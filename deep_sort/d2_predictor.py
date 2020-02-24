import argparse
import logging
import os
from collections import OrderedDict
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)

from detectron2.modeling import build_model
from detectron2.utils import comm
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.logger import setup_logger

import torch.nn.functional as F

class TensorPredictor:
    """
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format
        
        self.short_edge_length = cfg.INPUT.MIN_SIZE_TEST
        self.max_size = cfg.INPUT.MAX_SIZE_TEST
        
    def _resize_shortest_edge(img):
        
        img_size = img.size()[1:]

        scale = self.short_edge_length * 1.0 / torch.min(img_size)
        if img_size[0] < img_size[1]:
            newh, neww = size, scale * img_size[1]
        else:
            newh, neww = scale * img_size[0], size
        if torch.max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / torch.max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return F.interpolate(img, newh, neww)

    def __call__(self, original_image):
        """
        """
        with torch.no_grad():
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = _resize_shortest_edge(original_image)
            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions
