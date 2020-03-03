import os
import cv2
import time
import argparse
import numpy as np
from distutils.util import strtobool

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

import torch

from deep_sort import DeepSort
from util import COLORS_10, draw_bboxes

import glob, json, shutil

number_of_iterations = 100
termination_eps = 0.00001
warp_mode = cv2.MOTION_EUCLIDEAN
warp_matrix = np.eye(2, 3, dtype=np.float32)
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

import logging

logging.basicConfig(level=logging.DEBUG, filename='/content/app.log', filemode='w')


class Detector(object):
    def __init__(self, args):
        self.args = args
        use_cuda = bool(strtobool(self.args.use_cuda))
        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if not args.image_input:
            self.vdo = cv2.VideoCapture()
        cfg = get_cfg()
        #cfg.merge_from_file("detectron2_repo/configs/COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml")
        #cfg.MODEL.WEIGHTS = "detectron2://COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x/139686956/model_final_5ad38f.pkl"
        cfg.merge_from_file("../detectron2_repo/configs/Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml")
        cfg.MODEL.WEIGHTS = args.detectron2_weights
        #"detectron2://Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv/18131413/model_0039999_e76410.pkl"
        cfg.MODEL.MASK_ON = False
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 
        #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.7

        self.predictor = DefaultPredictor(cfg)
        self.deepsort = DeepSort(args.deepsort_checkpoint, use_cuda=use_cuda)
        #self.class_names = self.yolo3.class_names

    def __enter__(self):
        if not args.image_input:
            assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
            self.vdo.open(self.args.VIDEO_PATH)
            assert self.vdo.isOpened()
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            self.img_list = sorted(glob.glob(os.path.join(self.args.VIDEO_PATH, "*")))
            img_test = cv2.imread(self.img_list[0])
            self.im_height, self.im_width = img_test.shape[:2]

        if self.args.save_path:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter(self.args.save_path, fourcc, args.save_fps, (self.im_width, self.im_height))
            
        if self.args.save_frames:
            if os.path.exists('supervisely'):
                import shutil
                shutil.rmtree('supervisely')
            os.makedirs('supervisely')
            os.makedirs('supervisely/img')
        
        if self.args.save_txt:
            self.txt = open('gt.txt', "w")
        
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def detect(self):
        
        start = time.time()
        if not args.image_input:            

            start_second = 0
            end_second = 8

            fps = self.vdo.get(cv2.CAP_PROP_FPS)

            print('fps: ', fps)

            start_frameid = start_second * fps
            end_frameid = end_second * fps
        else:
            frame_id = 0
            
        if self.args.update_tracks:
            shutil.copytree(self.args.detections_dir, self.args.detections_dir + '_tracked')
        
        while True:
            
            print(f'FRAME_ID: {frame_id}')
            logging.debug(f'FRAME_ID: {frame_id}')
            
            new_sequence = False
            
            if not args.image_input:                
                frame_id = int(round(self.vdo.get(1)))            
                if frame_id < start_frameid:
                    continue
                elif frame_id > end_frameid:
                    break           
                _, ori_im = self.vdo.read() # retrieve()
            else:
                if frame_id>=(len(self.img_list)):
                    break
                    
                if frame_id > 1:
                    prev_im = ori_im
                    
                ori_im = cv2.imread(self.img_list[frame_id])
                    
                if frame_id > 1:

                    im1_gray = cv2.cvtColor(prev_im, cv2.COLOR_RGB2GRAY)
                    im2_gray = cv2.cvtColor(ori_im, cv2.COLOR_RGB2GRAY)

                    cc, _ = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria, None, 1)
                    
                    new_sequence = cc > 0.5

                frame_id+=1                
            
            if self.args.save_frames:
                if not args.image_input:
                    cv2.imwrite(f'./supervisely/img/img_{frame_id:05}.jpg', ori_im)
                else:
                    cv2.imwrite(f'./supervisely/img/' + self.img_list[frame_id-1][-13:], ori_im)
            
            im = ori_im
            predictions = self.predictor(im)
            
            instances = predictions["instances"]

            if instances.pred_classes.numel() > 0:                

                #print(instances.pred_classes)
                
                mask = instances.pred_classes == 0

                scores = instances.scores[mask]
                pred_boxes = instances.pred_boxes[mask]

                xcyc = pred_boxes.get_centers()
                wh = pred_boxes.tensor[:, 2:] - pred_boxes.tensor[:, :2] + torch.ones(pred_boxes.tensor[:, 2:].size()).cuda()
                
                wh_min, _ = torch.min(wh, 1)            
                
                # if "pred_masks" in instances.keys():
                #	pred_masks = instances["pred_masks"][mask]

                bbox_xcycwh = torch.cat((xcyc, wh), 1)[wh_min >=4].detach().cpu().numpy()
                cls_conf = scores.detach().cpu().numpy()
                
                if self.args.detections_dir!="":
                    ann_dir = os.path.join(self.args.detections_dir)
                    
                    ann = os.path.basename(self.img_list[frame_id-1]) + ".json"
                    ann_path = os.path.join(ann_dir, 'MOT', 'ann', ann)
                    
                    with open(ann_path) as f:
                        ann_dict = json.load(f)
                    bboxes = []
                    for obj in ann_dict['objects']:
                        bbox = obj["points"]["exterior"]
                        bbox = bbox[0]+bbox[1]
                        bbox = [min(bbox[0], bbox[2]), min(bbox[1], bbox[3]), max(bbox[0], bbox[2]), max(bbox[1], bbox[3])]
                        bboxes.append([(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2, bbox[2]-bbox[0], bbox[3]-bbox[1]])
                        
                    bbox_xcycwh = np.array(bboxes)
                    cls_conf = np.ones(bbox_xcycwh.shape[0])
                
                #print(bbox_xcycwh, cls_conf)

                #bbox_xcycwh[:, 3:] *= 1.2

                outputs, detections = self.deepsort.update(bbox_xcycwh, cls_conf, im, new_sequence)
                if len(outputs) > 0:                    
                    bbox_xyxy = outputs[:, :4]
                    #dh = ((0.1/1.2)*(bbox_xyxy[:,3]-bbox_xyxy[:,1])).astype(int)
                    #bbox_xyxy[:,1] += dh
                    #bbox_xyxy[:,3] -= dh
                    identities = outputs[:, 4]
                    match_method = outputs[:, 5]
                    number = outputs[:, 6]
                    number_bbox = outputs[:, 7:]
                    ori_im = draw_bboxes(frame_id, ori_im, bbox_xyxy, identities, match_method, number, number_bbox)
                    
                    if self.args.save_txt:
                        for j in range(bbox_xyxy.shape[0]):
                            x1 = bbox_xyxy[j,0]
                            y1 = bbox_xyxy[j,1]
                            x2 = bbox_xyxy[j,2]
                            y2 = bbox_xyxy[j,3]
                            self.txt.write(f'{frame_id},{identities[j]},{x1},{y1},{x2-x1},{y2-y1},1,0,-1,-1\n')
                if self.args.update_tracks:                    
                    ann_path = os.path.join(self.args.detections_dir + '_tracked', 'MOT', 'ann', ann)
                    print(ann_path)
                    
                    for idx, obj in enumerate(ann_dict['objects']):
                        obj["tags"] = [{"name": "track_id", "value": detections[idx].track_id}]
                        
                    with open(ann_path, 'w') as f:
                        json.dump(ann_dict, f)

            end = time.time()
            print("time: {}s, fps: {}".format(end - start, 1 / (end - start)))

            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.output.write(ori_im)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--detectron2_cfg", type=str, default="YOLOv3/cfg/yolo_v3.cfg")
    parser.add_argument("--detectron2_weights", type=str, default="detectron2://Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv/18131413/model_0039999_e76410.pkl")
    parser.add_argument("--yolo_names", type=str, default="YOLOv3/cfg/coco.names")
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    parser.add_argument("--nms_thresh", type=float, default=0.4)
    parser.add_argument("--deepsort_checkpoint", type=str, default="deep_sort/deep/checkpoint/ckpt.t7")
    parser.add_argument("--max_dist", type=float, default=0.2)
    parser.add_argument("--ignore_display", dest="display", action="store_false")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="demo.avi")
    parser.add_argument("--use_cuda", type=str, default="True")
    parser.add_argument("--save_frames", action="store_true")
    parser.add_argument("--save_txt", action="store_true")
    parser.add_argument("--image_input", action="store_true")
    parser.add_argument("--save_fps", type=int, default=20)
    parser.add_argument("--detections_dir", type=str, default="")
    parser.add_argument("--update_tracks", action="store_true")
    
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with Detector(args) as det:
        det.detect()

