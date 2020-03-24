#json_file = '/content/output/inference/inference_coco.json'
#image_root = '/content/LhcTGafSgw_all'

from detectron2.data.datasets import load_coco_json
#results = load_coco_json(json_file, image_root, dataset_name='players')

#from detectron2.data.datasets import register_coco_instances
#register_coco_instances('players', {}, json_file, image_root)

import argparse
import copy
import numpy as np

from detectron2.data import detection_utils as utils
#from detectron2.data import transforms as T

class PlayerMapper:
    """
    """

    def __init__(self): # , cfg

        #self.tfm_gens = utils.build_transform_gen(cfg, False)
        self.img_format = "BGR" #cfg.INPUT.FORMAT

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        
        # todo: decide wheather dataset_dict is needed
        
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image) # ?

        image_list = []

        for ann in dataset_dict['annotations']:
          #bbox = BoxMode.convert(ann['bbox'], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
          bbox = [int(x) for x in ann['bbox']]
          image_crop = image[bbox[1]:(bbox[1]+bbox[3]),bbox[0]:(bbox[0]+bbox[2])]
          #image_crop, _ = T.apply_transform_gens(self.tfm_gens, image_crop)
          image_crop = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

          image_list.append({
              "image": image_crop,
              "height": bbox[3],
              "width": bbox[2],
              "file_name": dataset_dict["file_name"],
          })          

        #dataset_dict["image_list"] = image_list

        new_dict = {"image_list": image_list}

        return new_dict #dataset_dict


import torch

from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data import samplers

def trivial_batch_collator(batch):
    """
    """

    collated_list = []

    for img in batch:
      collated_list.extend(img['image_list'])

    return collated_list

def build_players_loader(json_file, image_root, batch_size):
    """
    """
    dataset_dicts = load_coco_json(json_file, image_root)

    dataset = DatasetFromList(dataset_dicts)
    dataset = MapDataset(dataset, PlayerMapper())

    sampler = samplers.InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers = batch_size,
        batch_sampler = batch_sampler,
        collate_fn = trivial_batch_collator,
    )
    return data_loader



def parse_args():
    parser = argparse.ArgumentParser()  
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--json_path", type=str)
    parser.add_argument("--batch_size", type=int)                        

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    players_loader = build_players_loader(args.json_path, args.image_dir, args.batch_size)
