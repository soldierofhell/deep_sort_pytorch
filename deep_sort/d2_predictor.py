import logging
import os
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog

from detectron2.modeling import build_model
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

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format
        
        self.min_size = cfg.INPUT.MIN_SIZE_TEST
        self.max_size = cfg.INPUT.MAX_SIZE_TEST
        
    def _resize_shortest_edge(img):
        
        img_size = img.size()[1:]

        scale = self.min_size * 1.0 / torch.min(img_size)
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

    def __call__(self, image_list):
        """
        """
        with torch.no_grad():
            input_list = []
            
            for image_tensor in image_list:
                # Apply pre-processing to image.
                if self.input_format == "RGB":
                    # whether the model expects BGR inputs or RGB
                    image_tensor = image_tensor[:, :, ::-1]
                height, width = image_tensor.size()[1:]
                    image = _resize_shortest_edge(image_tensor)
            
            
            input_list.append({"image": image, "height": height, "width": width})
            predictions = self.model(input_list)
            return predictions
