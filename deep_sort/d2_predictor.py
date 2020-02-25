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
        
    def _resize_shortest_edge(self, img):
        
        h, w = img.size()[1], img.size()[2]

        scale = self.min_size * 1.0 / min(h, w)
        if h < w:
            newh, neww = self.min_size, scale * w
        else:
            newh, neww = scale * h, self.min_size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return F.interpolate(torch.unsqueeze(img,0), newh, neww)[0]

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
                height, width = image_tensor.size()[1], image_tensor.size()[2]
                image = self._resize_shortest_edge(image_tensor)           
                input_list.append({"image": image, "height": height, "width": width})
                
            predictions = self.model(input_list)
            return predictions
