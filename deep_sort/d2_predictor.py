import logging
import os
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog

from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger

import torch.nn.functional as F

import time

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
        
        self.batch_size = int(cfg.IMAGES_PER_BATCH_TEST)
        
        #print('detectron parameters: ', self.input_format, self.min_size, self.max_size)
        
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
        return F.interpolate(torch.unsqueeze(img,0), (newh, neww))[0]

    def __call__(self, image_list):
        """
        """
        
        batch_list = list(range(len(image_list)))
        batch_list = [batch_list[i:i + self.batch_size] for i in range(0, len(image_list), self.batch_size)]
        
        with torch.no_grad():
            
            predictions = []
            
            for batch_ind in batch_list:
                tick = time.time()
                input_list = []                
                #image_batch = [image for (i, image) in enumerate(image_list) if i in batch_ind]
                for idx in batch_ind:
                    image_tensor = image_list[idx]
                    # Apply pre-processing to image.
                    if self.input_format == "RGB":
                        # whether the model expects BGR inputs or RGB
                        image_tensor = image_tensor[:, :, ::-1]
                    height, width = image_tensor.size()[1], image_tensor.size()[2]
                    image = self._resize_shortest_edge(image_tensor)
                    image = image * 256.0 #.permute(2, 0, 1)
                    #print('image size: ', image.size())
                    #print('height, width: ', height, width)
                    input_list.append({"image": image, "height": height, "width": width})
                    
                print('preprocessing: ', time.time() - tick)
                
                tick = time.time()
                predictions.extend(self.model(input_list))
                print('forward: ', time.time() - tick)
                
            return predictions
