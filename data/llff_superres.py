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

def get_dstype(s):
    return s.split("/")[-3]

class LLFFSuperResDataset(SuperResDataset):
    def load_paths_dstype(self, cfg, dstype):
        base_path = f"/nobackup/nvme1/datasets/llff/{dstype}/images/" # replace with your path
        files = list(glob(base_path + "*.jpg")) + list(glob(base_path + "*.JPG"))
        files = list(sorted(files))

        frameskip = cfg.model.frameskip
        for a, b, c in zip(files[:-2*frameskip],files[frameskip:-frameskip],files[2*frameskip:]):
            self.frame_paths.append([a, b, c])

    def load_paths(self, cfg, split):
        if cfg.dataset.dstype == "all":
            assert self.split in ['training', 'validation'], "Split must be validation"
        else:
            assert self.split in ['validation'], "Split must be validation"
        
        self.frame_paths = []
        self.flow_paths = None
        self.mask_paths = None
        split_path = "data/splits/llff/llff_split.py"
        if cfg.dataset.dstype == "all":
            dslist = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
            for dstype in dslist:
                self.load_paths_dstype(cfg, dstype)
        else:
            self.load_paths_dstype(cfg, cfg.dataset.dstype)

        with open("data/splits/llff/llff_split.py", "r") as f:
            lines = f.readlines()
            split_map = {l[:l.index(",")]: l[l.index(",")+1:-1] for l in lines}
        with open("data/splits/llff/llff_max.txt", "r") as f:
            flow_max = {}
            for l in f.readlines():
                flow_max[l[:l.index(",")]] = float(l[l.index(",")+1:])

        clean_frame_paths = []
        for seq in self.frame_paths:
            if split_map[seq[1]] != self.split:
                continue
            if cfg.dataset.flow_max > 0 and flow_max[seq[1]] >= cfg.dataset.flow_max * cfg.dataset.imsz_super:
                continue
            clean_frame_paths.append(seq)
        self.frame_paths = clean_frame_paths

        print("Dataset:", self.split, len(self.frame_paths))
