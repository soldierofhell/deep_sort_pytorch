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

import logging
logging.basicConfig(level=logging.DEBUG, filename='/content/app.log', filemode='w')

__all__ = ['DeepSort']

# todo:
# * usunac xywh, zostawić xyxy?

# * parametry: players_list_path, game_id, ref_img_paths (format gameid_team.jpg), checkpoint_paths, 


class DeepSort(object):
    def __init__(self, model_path, max_dist=0.2, use_cuda=True, extractor_type='pedestrian', game_id=0, team_0='Belgium'):
        self.min_confidence = 0.5
        #self.nms_max_overlap = 1.0

        self.extractor_type = extractor_type
        if self.extractor_type == 'pedestrian':
            self.extractor = Extractor(model_path, use_cuda=use_cuda)
        else:
            self.extractor = SimilarityPredictor('/content/players_ckpt.pth')

        max_cosine_distance = max_dist
        nn_budget = 100
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        
        
        number_cfg = get_cfg()
        number_cfg.merge_from_file("/content/detectron2_repo/configs/Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml")
        number_cfg.MODEL.WEIGHTS = "/content/model_0036999.pth"
        number_cfg.MODEL.MASK_ON = False
        number_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        number_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.number_detector = ObjectDetector(number_cfg)        
        
        self.number_decoder = TextPredictor()
        
        self.game_id = game_id
        self.team_0 = team_0
        self.players_list = []
        self.team_numbers = [[],[]]
        import csv
        with open('/content/datasets.csv') as csvfile:
          reader = csv.DictReader(csvfile)
          for row in reader:
            if row['GAME_ID'] == str(self.game_id):
                self.players_list.append(row)
                if row['TEAM'] == self.team_0:
                    self.team_numbers[0].append(row['NUMBER'])
                else:
                    self.team_numbers[1].append(row['NUMBER'])        
            
        
        
        self.team_embeddings = SimilarityPredictor('/content/teams_ckpt.pth')
        #self.team_threshold = 0.75
        team_ref_paths = ['/content/team0_ref.jpg', '/content/team1_ref.jpg']
        team_ref_img = [TF.to_tensor(cv2.imread(path)).cuda() for path in team_ref_paths]
        self.team_ref_embeddings = self.team_embeddings.predict(team_ref_img)
        
        self.tracker = Tracker(metric, team_numbers=self.team_numbers)
        
        self.track_history = {}
        self.detection_history = {}

    def update(self, bbox_xywh, confidences, ori_img, new_sequence, frame_id, img_name):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        if self.extractor_type == 'pedestrian':
            features = self._get_features(bbox_xywh, ori_img)
            numbers, team_ids = self._predict_numbers(bbox_xywh, ori_img)
        else:
            numbers, team_ids, features = self._predict_numbers(bbox_xywh, ori_img)
        # TODO: change this name           
            
       
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        
        #temp_number = {'number': None, 'confidence': None} # numbers[i]
        self.detections = [Detection(bbox_tlwh[i], conf, features[i], numbers[i], team_ids[i]) for i,conf in enumerate(confidences) if conf>self.min_confidence]      
        
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

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if track.is_deleted() or track.time_since_update > 0:
                continue
            box = track.to_tlwh()
            x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            match_method = track.match_method
            number = track.number if track.number is not None else -1
            number_bbox = track.number_bbox if track.number_bbox is not None else [0,0,0,0]
            detection_id = track.detection_id
            min_cost = track.min_cost
            outputs.append(np.array([x1,y1,x2,y2,track_id, match_method, number, number_bbox[0],number_bbox[1],number_bbox[2],number_bbox[3], detection_id, min_cost], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs,axis=0)
        return outputs, self.detections


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
    
    def _valid_box(self, number_bbox, player_bbox):
        
        player_area = (player_bbox[2] - player_bbox[0]) * (player_bbox[3] - player_bbox[1])
        number_area = (number_bbox[2] - number_bbox[0]) * (number_bbox[3] - number_bbox[1])
        
        number_y_center = (number_bbox[3]+number_bbox[1])/(2*(player_bbox[3] - player_bbox[1]))

        height_ratio = (number_bbox[3]-number_bbox[1])/(player_bbox[3] - player_bbox[1])
                   
        valid_box = number_area/player_area > 0.02 and number_y_center > 0.2 and number_y_center < 0.4 and height_ratio > 0.15
        
        return valid_box
            
    
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
                'team_id': track.team_id,
            }
            track_list.append(track_dict)           
        self.track_history[img_name] = track_list
        
        detection_list = []
        for idx, detection in enumerate(self.detections):
            detection_dict = {
                'detection_id': idx,
                'bbox': self._tlwh_to_xyxy(detection.tlwh),
                'confidence': detection.confidence,
            }
            detection_list.append(detection_dict)           
        self.detection_history[img_name] = detection_list
            

    def export(self, export_dir):  
        with open(os.path.join(export_dir, 'tracks.json'), 'w') as f:
            json.dump(self.track_history, f)
        with open(os.path.join(export_dir, 'detections.json'), 'w') as f:
            json.dump(self.detection_history, f)
        with open(os.path.join(export_dir, 'matched.json'), 'w') as f:
            json.dump(self.tracker.matched_tracks, f)



