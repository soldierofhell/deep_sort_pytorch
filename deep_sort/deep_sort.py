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

import torchvision.transforms.functional as TF

import cv2

__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, model_path, max_dist=0.2, use_cuda=True):
        self.min_confidence = 0.6
        self.nms_max_overlap = 1.0

        self.extractor = Extractor(model_path, use_cuda=use_cuda)

        max_cosine_distance = max_dist
        nn_budget = 100
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)
        
        number_cfg = get_cfg()
        number_cfg.merge_from_file("/content/detectron2_repo/configs/Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml")
        number_cfg.MODEL.WEIGHTS = "/content/model_0029999.pth"
        number_cfg.MODEL.MASK_ON = False
        number_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        number_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.number_detector = ObjectDetector(number_cfg)        
        
        self.number_decoder = TextPredictor()
        
        self.game_id = 0
        self.players_list = []
        self.team_numbers = [[],[]]
        import csv
        with open('/content/datasets.csv') as csvfile:
          reader = csv.DictReader(csvfile)
          for row in reader:
            if row['GAME_ID'] == str(self.game_id):
                self.players_list.append(row)
                if row['TEAM'] == 'Belgium':
                    self.team_numbers[0].append(row['NUMBER'])
                else:
                    self.team_numbers[1].append(row['NUMBER'])        
            
        self.team_embeddings = SimilarityPredictor('/content/teams_ckpt.pth')
        #self.team_threshold = 0.75
        team_ref_paths = ['/content/team0_ref.jpg', '/content/team1_ref.jpg']
        team_ref_img = [TF.to_tensor(cv2.imread(path)).cuda() for path in team_ref_paths]
        self.team_ref_embeddings = self.team_embeddings.predict(team_ref_img)

    def update(self, bbox_xywh, confidences, ori_img):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_xywh, ori_img)
        numbers = self._predict_numbers(bbox_xywh, ori_img)
        
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        
        #temp_number = {'number': None, 'confidence': None} # numbers[i]
        detections = [Detection(bbox_tlwh[i], conf, features[i], numbers[i]) for i,conf in enumerate(confidences) if conf>self.min_confidence]      
        
        # run on non-maximum supression
        #boxes = np.array([d.tlwh for d in detections])
        #scores = np.array([d.confidence for d in detections])
        #indices = non_max_suppression( boxes, self.nms_max_overlap, scores)
        #detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if track.is_deleted() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            match_method = track.match_method
            number = track.number if track.number is not None else -1
            number_bbox = track.number_bbox if track.number_bbox is not None else [0,0,0,0]
            outputs.append(np.array([x1,y1,x2,y2,track_id, match_method, number, number_bbox[0],number_bbox[1],number_bbox[2],number_bbox[3]], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs,axis=0)
        return outputs, detections


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

        bbox[0] = max(bbox[0]-int(0.1*bw), 0)
        bbox[1] = max(bbox[1]-int(0.1*bh), 0)
        bbox[2] = min(bbox[2]+int(0.1*bw), w)
        bbox[3] = min(bbox[3]+int(0.1*bh), h)

        return bbox
    
    def _predict_numbers(self, bbox_xywh, ori_img):
        
       
        # todo: 100% CUDA
        
        crop_list = []
        
        for box in bbox_xywh:
            x1,y1,x2,y2 = self._xywh_to_xyxy(box)
            player_crop = ori_img[y1:y2,x1:x2]
            crop_list.append(TF.to_tensor(player_crop).cuda())
            
        # split to teams
        embeddings = self.team_embeddings(crop_list)
        dists = torch.cdist(embeddings, self.team_ref_embeddings)
        team_ids = torch.argmin(dists, dim=1)
        
        # number detection
        number_instances = self.number_detector(crop_list)
        
        numbers = []
        for team_id, number_instance in zip(team_ids, number_instances):
            if number_instance.pred_classes.size()[0]>0:
                number_box = number_instance.pred_boxes.tensor[0].detach().cpu().numpy().astype(int)
                padded_box = self._padded_bbox(number_box, player_crop.shape[0], player_crop.shape[1])     
                number_crop = player_crop[padded_box[1]:padded_box[3], padded_box[0]:padded_box[2]]

                pred, confidence_score = self.number_decoder.predict(number_crop, input_size=(100, 32), dictionary=self.team_numbers[i])
                
                
                numbers.append({'number': pred, 'confidence': confidence_score, 'bbox': number_box.tolist()})
            else:
                numbers.append({'number': None, 'confidence': None, 'bbox': None})
        
        return numbers
    



