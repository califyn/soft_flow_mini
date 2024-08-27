import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from glob import glob

from .superres import SuperResDataset

def frame_to_fwd_flow(s):
    return s.replace("image_2", "flow_occ")

class KITTISuperResDataset(SuperResDataset):
    def load_paths(self, cfg, split):
        assert self.split in ['training', 'validation', 'all'], "Split must be training or validation"
        self.collate_different_shapes_mode = "first"

        base_path = f"/data/scene-rep/custom/KITTI/KITTI/training/" # replace with your path
        split_path = "data/splits/kitti/kitti_split.py"
        frameskip = cfg.model.frameskip
        
        self.frame_paths = []
        self.flow_paths = []
        files = [
            list(glob(base_path + f"image_2/*_{(10-frameskip):02d}.png")),
            list(glob(base_path + "image_2/*_10.png")),
            list(glob(base_path + f"image_2/*_{(10+frameskip):02d}.png")),
        ]
        files = [list(sorted(files_sub)) for files_sub in files]
        with open(split_path, "r") as f:
            lines = f.readlines()
            split_map = {l[:l.index(",")]: l[l.index(",")+1:-1] for l in lines}

            for a, b, c in zip(*files):
                if b not in split_map:
                    raise ValueError
                if self.split == 'all' or split_map[b] == self.split:
                    self.frame_paths.append([a, b, c])
                    self.flow_paths.append([frame_to_fwd_flow(b)])

        self.mask_paths = [{}] * len(self)
        print("Dataset:", self.split, len(self.frame_paths))
