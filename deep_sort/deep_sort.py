import numpy as np

from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
#from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker

from detectron2.config import get_cfg
#from detectron2.engine import DefaultPredictor
from .d2_predictor import TensorPredictor as ObjectDetector
from .deep_text_recognition_benchmark.text_predictor import TextPredictor
from .similarity.predictor import TensorPredictor as SimilarityPredictor

import torch
import torchvision.transforms.functional as TF

import cv2
import json
import os

import tqdm
import time

import pickle
import glob

import numpy as np
import pandas as pd

from respomodules.pitch_geom.homography import get_pitch_homography
from respoml.core.modules.pitch_geom.hom_utils import warp_points_H


import logging
logging.basicConfig(level=logging.DEBUG, filename='/content/app.log', filemode='w')

__all__ = ['DeepSort']

# todo:
# * usunac xywh, zostawić xyxy?

# * parametry: players_list_path, game_id, ref_img_paths (format gameid_team.jpg), checkpoint_paths, 

import configparser
from .players import build_players_loader


class DeepSort(object):
    def __init__(self, config_path):
        
        config = configparser.ConfigParser()
        config.read(config_path)
        
        self.min_confidence = 0.5
        #self.nms_max_overlap = 1.0
        
        self.img_list = sorted(glob.glob(os.path.join(config['input']['image_dir'], "*")))
        # TODO: check first image
        self.image_width = config['input'].getint('image_width')
        self.image_height = config['input'].getint('image_height')
        
        self.ecc_threshold = config['sequence_detection'].getfloat('ecc_threshold')
        
        # player detections
        
        # TODO: filter corrupted detections
        
        self.detections_path = config['output']['detections_path']
        
        if config['player_detections'].getboolean('use_gt'):
            json_path = config['player_detections']['gt_json']
        else:
            json_path = config['player_detections']['predicted_json']       
  
        
        self.players_loader = build_players_loader(json_path, config['input']['image_dir'], config['player_detections'].getint('batch_size'))        

        # player reid
        
        metric = NearestNeighborDistanceMetric(config['player_reid']['metric'], config['player_reid'].getfloat('max_distance'), config['player_reid'].getint('budget'))
        
        self.extractor_type = config['player_reid']['extractor_type']
        if self.extractor_type == 'pedestrian':
            self.extractor = Extractor(config['player_reid']['checkpoint'], use_cuda=True)
        else:
            self.extractor = SimilarityPredictor(config['player_reid']['checkpoint'])
            
        # team reid
        
        self.team_embeddings = SimilarityPredictor(config['team_reid']['checkpoint'])
        
        # number detection
        
        self.number_enabled = config['number_detection'].getboolean('enabled')
        
        if self.number_enabled:
        
            number_cfg = get_cfg()
            number_cfg.merge_from_file(config['number_detection']['cfg'])
            number_cfg.MODEL.WEIGHTS = config['number_detection']['checkpoint']
            number_cfg.MODEL.MASK_ON = False
            number_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
            number_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config['number_detection'].getfloat('detection_threshold')

            number_cfg.IMAGES_PER_BATCH_TEST = config['number_detection'].getint('batch_size')

            self.number_detector = ObjectDetector(number_cfg)

            # number recognition

            self.number_decoder = TextPredictor(config['number_recognition']['checkpoint'])
        
        # external data
        
        #self.game_id = game_id
        #self.team_0 = team_0
        
        self.players_list = []
        self.team_numbers = [[],[]]
        import csv
        with open(config['game_data']['game_csv']) as csvfile:
          reader = csv.DictReader(csvfile)
          for row in reader:
            if row['GAME_ID'] == str(config['game_data']['game_id']):
                self.players_list.append(row)
                if row['TEAM'] == config['game_data']['team_0_name']:
                    self.team_numbers[0].append(row['NUMBER'])
                else:
                    self.team_numbers[1].append(row['NUMBER'])       

        #self.team_threshold = 0.75
        team_ref_paths = [config['game_data']['team_0_photo'], config['game_data']['team_1_photo']]
        team_ref_img = [TF.to_tensor(cv2.imread(path)).cuda() for path in team_ref_paths]
        self.team_ref_embeddings = self.team_embeddings.predict(team_ref_img)
        
        # position: pandas df -> dict={image_name: h=np.array(3,3)}       
        
        hom_df = pd.read_csv(config['position']['homeography_csv'])
        self.hom_dict = hom_df.set_index('frame_index').iloc[:,:9].T.to_dict('list')
        for k, v in self.hom_dict.items():
            self.hom_dict[k] = np.array(v).reshape((3,3))
        self.hom_width = config['position'].getint('input_width')
        self.hom_height = config['position'].getint('input_height')
        
        # tracking
        
        self.tracker = Tracker(metric, team_numbers=self.team_numbers)
        
        self.track_history = {}
        self.detection_history = {}

    def _player_coordinates(self, X, h):
        H = get_pitch_homography(h, (self.image_height, self.image_width), orig_size=(320, 640))
 
        return warp_points_H(X, H)
    
    def update(self, bbox_xywh, confidences, features, numbers, team_ids, categories, ori_img, new_sequence, frame_id, img_name):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        #if self.extractor_type == 'pedestrian':
        #    features = self._get_features(bbox_xywh, ori_img)
        #    numbers, team_ids = self._predict_numbers(bbox_xywh, ori_img)
        #else:
        #    numbers, team_ids, features = self._predict_numbers(bbox_xywh, ori_img)
        # TODO: change this name
        
        
        hom_key = os.path.basename(self.img_list[frame_id])[:-4]
        h = self.hom_dict[hom_key]
        X = np.stack((bbox_xywh[:,0]+bbox_xywh[:,2]/2.0, bbox_xywh[:,1]+bbox_xywh[:,3]), axis=1)
        #print(h, X)
        player_coordinates = self._player_coordinates(X, h)
       
        bbox_tlwh = bbox_xywh # self._xywh_to_tlwh(bbox_xywh)
        
        #temp_number = {'number': None, 'confidence': None} # numbers[i]
        self.detections = [Detection(bbox_tlwh[i],
                                     conf,
                                     features[i],
                                     numbers[i],
                                     team_ids[i],
                                     categories[i],
                                     player_coordinates[i]
                                    ) for i,conf in enumerate(confidences) if conf>self.min_confidence]      
        
        # run on non-maximum supression
        #boxes = np.array([d.tlwh for d in detections])
        #scores = np.array([d.confidence for d in detections])
        #indices = non_max_suppression( boxes, self.nms_max_overlap, scores)
        #detections = [detections[i] for i in indices]

        # update tracker
        if new_sequence:
            self.tracker.update_numbers()
            logging.debug(self.tracker.matched_numbers)
            
        self.tracker.predict()
        self.tracker.update(self.detections, new_sequence)
        
        self._add_frame_history(img_name)

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        bbox_xywh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
        bbox_xywh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
        return bbox_xywh


    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),self.width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),self.height-1)
        return x1,y1,x2,y2
    
    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1,y1,x2,y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2,x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features
    
    def _padded_bbox(self, bbox, h, w):
        bw, bh = bbox[2]-bbox[0], bbox[3]-bbox[1]
        
        padded_bbox = bbox.copy()

        padded_bbox[0] = max(bbox[0]-int(0.1*bw), 0)
        padded_bbox[1] = max(bbox[1]-int(0.1*bh), 0)
        padded_bbox[2] = min(bbox[2]+int(0.1*bw), w)
        padded_bbox[3] = min(bbox[3]+int(0.1*bh), h)

        return padded_bbox
    
    def _valid_box(self, number_bbox, h, w):
        
        player_area = w * h
        number_area = (number_bbox[2] - number_bbox[0]) * (number_bbox[3] - number_bbox[1])
        
        number_y_center = (number_bbox[3]+number_bbox[1])/(2*h)

        height_ratio = (number_bbox[3]-number_bbox[1])/h
                   
        valid_box = number_area/player_area > 0.02 and number_y_center > 0.2 and number_y_center < 0.4 and height_ratio > 0.15
        
        return valid_box
            
    def forward_tracking(self):
        number_of_iterations = 100
        termination_eps = 0.00001
        warp_mode = cv2.MOTION_EUCLIDEAN
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
        

        img_test = cv2.imread(self.img_list[0])
        self.im_height, self.im_width = img_test.shape[:2]
        
        for frame_id, image_path in tqdm.tqdm(enumerate(self.img_list)):  
            
            new_sequence = False
            
            #image_file
            
                   
            if frame_id > 0:
                prev_im = ori_im
                    
            ori_im = cv2.imread(self.img_list[frame_id])
            
            if frame_id > 0:

                im1_gray = cv2.cvtColor(prev_im, cv2.COLOR_RGB2GRAY) # todo: redundant
                im2_gray = cv2.cvtColor(ori_im, cv2.COLOR_RGB2GRAY)

                cc, _ = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria, None, 1)

                new_sequence = cc < self.ecc_threshold
                logging.debug(f'ECC: {cc}')
                
            im = ori_im # ?
            
            self.import_detections()
            
            detections = self.detections_dict[image_path]
            
            bbox_xcycwh = np.zeros((len(detections), 4))
            cls_conf = np.ones(len(detections)) # TODO: fill it with real numbers
            features = np.zeros((len(detections), 128))
            numbers = []
            team_ids = []
            categories = []
            
            
            for idx, detection in enumerate(self.detections_dict[image_path]):
                bbox_xcycwh[idx,:] = np.array(detection['bbox'])
                features[idx,:] = detection['features']
                numbers.append(detection['number'])
                team_ids.append(detection['team_id'])
                categories.append(detection['category_id'])
            
            # bbox_xcycwh = torch.cat((xcyc, wh), 1)[wh_min >=4].detach().cpu().numpy()
            # cls_conf
            
            # TODO: update(dict)
            self.update(bbox_xcycwh, cls_conf, features, numbers, team_ids, categories, im, new_sequence, frame_id, self.img_list[frame_id])
    
    
    def import_detections(self):
        
        with open(self.detections_path, 'rb') as f:
            self.detections_dict = pickle.load(f)
    
    def export_detections(self):
        
        # TODO: get rid of those lists.. ?
        file_names_all = []
        box_ids_all = []
        bboxes_all = []
        categories_all = []
        
        features_all = []
        team_ids_all = []
        numbers_all = []
        
        with torch.no_grad():
            for idx, input_list in tqdm.tqdm(enumerate(self.players_loader)):
                
                
                
                #tick = time.time()
                
                file_names_all.extend([input["file_name"] for input in input_list])
                box_ids_all.extend([input["box_id"] for input in input_list])
                bboxes_all.extend([input["bbox"] for input in input_list])
                categories_all.extend([input["category_id"] for input in input_list])
                
                crop_list = [input['image'] for input in input_list]
                
                #print('init processing: ', time.time()-tick)
                # player reid
                
                
                #tick = time.time()                
                
                if self.extractor_type != 'pedestrian':        
                    features = self.extractor.predict(crop_list).cpu().numpy()
                    features = [features[idx] for idx in range(features.shape[0])]
                    features_all.extend(features)
                    
                #print('player reid: ', time.time()-tick)
            
                # team reid
                
                #tick = time.time() 
                
                embeddings = self.team_embeddings.predict(crop_list)
                dists = torch.cdist(embeddings, self.team_ref_embeddings)        
                team_ids = torch.argmin(dists, dim=1).cpu().numpy().tolist()

                # todo: judge, goalkeeper
                del embeddings
                del dists
                
                #print('team reid: ', time.time()-tick)

                #print('team_ids: ', team_ids)

                #for crop in crop_list:
                    #logging.debug('crop size: ', torch.tensor(crop.size()).cpu().numpy().tolist())
                    #print(torch.tensor(crop.size()).cpu().numpy().tolist())

                # number detection
                
                if self.number_enabled:
                
                    number_outputs = self.number_detector(crop_list)

                    print('number detection: ', time.time()-tick)

                    #print('input length: ', len(crop_list))
                    #print('output length: ', len(number_outputs))

                    #tick = time.time()

                    numbers = []
                    for team_id, number_output, player_crop in zip(team_ids, number_outputs, crop_list):
                        number_instance = number_output['instances']
                        #logging.debug('detected boxes: ', number_instance.pred_classes.size()[0])
                        #print('detected boxes: ', torch.tensor(number_instance.pred_classes.size())[0].numpy().tolist())

                        if number_instance.pred_classes.size()[0]>0:
                            number_box = number_instance.pred_boxes.tensor[0].detach().cpu().numpy().astype(int)

                            if self._valid_box(number_box, player_crop.shape[1], player_crop.shape[2]):                    
                                padded_box = self._padded_bbox(number_box, player_crop.shape[1], player_crop.shape[2])     
                                #print('player crop: ', player_crop.size())
                                #print('number_box: ', number_box)
                                #print('padded_box: ', padded_box)
                                #print('tests :', number_instance.pred_boxes.tensor[0])
                                number_crop = player_crop[:, padded_box[1]:padded_box[3], padded_box[0]:padded_box[2]]

                                pred, confidence_score = self.number_decoder.predict(number_crop, input_size=(100, 32), dictionary=self.team_numbers[team_id])


                                numbers.append({'number': pred, 'confidence': confidence_score, 'bbox': number_box.tolist()})
                            else:
                                numbers.append({'number': None, 'confidence': None, 'bbox': None})
                        else:
                            numbers.append({'number': None, 'confidence': None, 'bbox': None})
                            
                else:
                    numbers = []
                    for idx in range(len(crop_list)):
                        numbers.append({'number': None, 'confidence': None, 'bbox': None})
                        
                #print('number recognition: ', time.time()-tick)

                numbers_all.extend(numbers)
                team_ids_all.extend(team_ids)


            #print('number dict: ', numbers_all)

            if self.extractor_type != 'pedestrian':
                out_dict = {}
                for idx in range(len(file_names_all)):
                    out_dict.setdefault(file_names_all[idx], []).append({
                        'box_id': box_ids_all[idx],
                        'bbox': bboxes_all[idx],
                        'number': numbers_all[idx],
                        'team_id': team_ids_all[idx],
                        'features': features_all[idx],
                        'category_id': categories_all[idx],
                    })
                        
                #np.save(self.detections_path, out_dict)
                with open(self.detections_path, 'wb') as f:
                    pickle.dump(out_dict, f)
    
    def _predict_numbers(self, bbox_xywh, ori_img):
        
       
        # todo: 100% CUDA
        
        batch_size = 4
        batch_list = list(range(len(bbox_xywh)))
        batch_list = [batch_list[i:i + batch_size] for i in range(0, len(batch_list), batch_size)]
                          
        numbers_all = []
        team_ids_all = []
        features_all = []
        
        for batch_ind in batch_list: 
        
            crop_list = []
            
            bbox_list = [self._xywh_to_xyxy(bbox) for (i, bbox) in enumerate(bbox_xywh) if i in batch_ind]

            h, w = ori_img.shape[:2]
            
            for bbox in bbox_list:
                x1,y1,x2,y2 = bbox
                bbox_w = x2-x1
                if bbox_w <10:
                    if x2+(10-bbox_w) < w:
                        x2 += 10-bbox_w
                    else:
                        x1 -= 10-bbox_w
                player_crop = ori_img[y1:y2,x1:x2]
                crop_list.append(TF.to_tensor(player_crop).cuda())

            # features
            
            if self.extractor_type != 'pedestrian':        
                features = self.extractor.predict(crop_list)
                features_all.extend(features)
            
            # split to teams
            embeddings = self.team_embeddings.predict(crop_list)
            dists = torch.cdist(embeddings, self.team_ref_embeddings)        
            team_ids = torch.argmin(dists, dim=1).cpu().numpy().tolist()

            # todo: judge, goalkeeper
            del embeddings
            del dists
        
            #print('team_ids: ', team_ids)
        
            #for crop in crop_list:
                #logging.debug('crop size: ', torch.tensor(crop.size()).cpu().numpy().tolist())
                #print(torch.tensor(crop.size()).cpu().numpy().tolist())
        
                
            # number detection
            number_outputs = self.number_detector(crop_list)

            #print('input length: ', len(crop_list))
            #print('output length: ', len(number_outputs))

            numbers = []
            for team_id, number_output, player_crop, player_bbox in zip(team_ids, number_outputs, crop_list, bbox_list):
                number_instance = number_output['instances']
                #logging.debug('detected boxes: ', number_instance.pred_classes.size()[0])
                #print('detected boxes: ', torch.tensor(number_instance.pred_classes.size())[0].numpy().tolist())

                if number_instance.pred_classes.size()[0]>0:
                    number_box = number_instance.pred_boxes.tensor[0].detach().cpu().numpy().astype(int)
                    
                    if self._valid_box(number_box, player_bbox):                    
                        padded_box = self._padded_bbox(number_box, player_crop.shape[1], player_crop.shape[2])     
                        #print('player crop: ', player_crop.size())
                        #print('number_box: ', number_box)
                        #print('padded_box: ', padded_box)
                        #print('tests :', number_instance.pred_boxes.tensor[0])
                        number_crop = player_crop[:, padded_box[1]:padded_box[3], padded_box[0]:padded_box[2]]

                        pred, confidence_score = self.number_decoder.predict(number_crop, input_size=(100, 32), dictionary=self.team_numbers[team_id])
              

                        numbers.append({'number': pred, 'confidence': confidence_score, 'bbox': number_box.tolist()})
                    else:
                        numbers.append({'number': None, 'confidence': None, 'bbox': None})
                else:
                    numbers.append({'number': None, 'confidence': None, 'bbox': None})
                
       
            numbers_all.extend(numbers)
            team_ids_all.extend(team_ids)

            
        #print('number dict: ', numbers_all)
        
        if self.extractor_type == 'pedestrian':        
            return numbers_all, team_ids_all
        else:
            features_all = np.array([f.cpu().numpy() for f in features_all])
            return numbers_all, team_ids_all, features_all
    
    
    def _add_frame_history(self, img_name):
        track_list = []
        for track in self.tracker.tracks:
            track_dict = {
                'track_id': track.track_id,
                'detection_id': track.detection_id,
                'state': track.state,
                'time_since_update': track.time_since_update,
                'kalman_box': self._tlwh_to_xyxy(track.to_tlwh()),
                'detection_box': self._tlwh_to_xyxy(track.detection.tlwh),
                'number': track.number if track.number is not None else 0,
                'number_bbox': track.number_bbox if track.number_bbox is not None else [0,0,0,0],
                'min_cost': track.min_cost,
                'detection_conf': track.detection.confidence,
                'detection_coordinates': track.detection.coordinates.tolist(),
                'team_id': track.team_id,
            }
            track_list.append(track_dict)           
        self.track_history[img_name] = track_list
        
        detection_list = []
        for idx, detection in enumerate(self.detections):
            
            # or from detection.__dict__
            
            detection_dict = {
                'detection_id': idx,
                'bbox': self._tlwh_to_xyxy(detection.tlwh),
                'confidence': detection.confidence,
                'category_id': detection.category_id,
                'coordinates': detection.coordinates.tolist(),
            }
            detection_list.append(detection_dict)           
        self.detection_history[img_name] = detection_list
            

    def export(self, export_dir):  
        with open(os.path.join(export_dir, 'tracks.json'), 'w') as f:
            json.dump(self.track_history, f)
        with open(os.path.join(export_dir, 'detections_2.json'), 'w') as f:
            json.dump(self.detection_history, f)
        with open(os.path.join(export_dir, 'matched.json'), 'w') as f:
            json.dump(self.tracker.matched_tracks, f)



