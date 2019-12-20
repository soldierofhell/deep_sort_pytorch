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


class Detector(object):
    def __init__(self, args):
        self.args = args
        use_cuda = bool(strtobool(self.args.use_cuda))
        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        self.vdo = cv2.VideoCapture()
        cfg = get_cfg()
        cfg.merge_from_file("detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
        self.predictor = DefaultPredictor(cfg)
        self.deepsort = DeepSort(args.deepsort_checkpoint, use_cuda=use_cuda)
        #self.class_names = self.yolo3.class_names

    def __enter__(self):
        assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
        self.vdo.open(self.args.VIDEO_PATH)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.args.save_path:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter(self.args.save_path, fourcc, 20, (self.im_width, self.im_height))

        assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def detect(self):
        while self.vdo.grab():
            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = ori_im
            predictions = self.predictor(im)
            
            instances = predictions["instances"]

            if instances.pred_classes.numel() > 0:                

                mask = instances.pred_classes == 1

                scores = instances.scores[mask]
                pred_boxes = instances.pred_boxes[mask]

                xcyc = pred_boxes.get_centers()
                wh = pred_boxes.tensor[:, :2] - pred_boxes.tensor[:, 2:]

                # if "pred_masks" in instances.keys():
                #	pred_masks = instances["pred_masks"][mask]

                bbox_xcycwh = torch.cat((xcyc, wh), 1).detach().cpu().numpy()
                cls_conf = scores.detach().cpu().numpy()
                
                print(bbox_xcycwh, cls_conf)

                bbox_xcycwh[:, 3:] *= 1.2

                outputs = self.deepsort.update(bbox_xcycwh, cls_conf, im)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    ori_im = draw_bboxes(ori_im, bbox_xyxy, identities)

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
    parser.add_argument("--detectron2_weights", type=str, default="YOLOv3/yolov3.weights")
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with Detector(args) as det:
        det.detect()

