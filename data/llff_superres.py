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

class LLFFSuperResDataset(SuperResDataset):
    def load_paths(self, cfg, split):
        assert self.split in ['validation'], "Split must be validation"

        base_path = f"/nobackup/nvme1/datasets/llff/{cfg.dataset.dstype}/images/" # replace with your path
        
        self.frame_paths = []
        self.flow_paths = None
        files = list(glob(base_path + "*.jpg"))
        files = list(sorted(files))
        frameskip = cfg.model.frameskip
        for a, b, c in zip(files[:-2*frameskip],files[frameskip:-frameskip],files[2*frameskip:]):
            self.frame_paths.append([a, b, c])

        print("Dataset:", self.split, len(self.frame_paths))
