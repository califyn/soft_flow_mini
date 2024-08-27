import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
import random
from glob import glob

from .superres import SuperResDataset

def get_scene(s, base_path):
    s = s[len(base_path):]
    s = s[:s.index("/")]
    return s

def frame_to_fwd_flow(s):
    s = s.replace("frame_", "flow_FW_")
    #s = s.replace(".png", ".npy")
    s = s.replace(".png", ".flo5")
    return s

class SpringSuperResDataset(SuperResDataset):
    def __init__(self, cfg, split):
        super().__init__(cfg, split)
        self.flow_multiplier = 2

    def load_paths(self, cfg, split):
        assert self.split in ['training', 'validation', 'all'], "Split must be training or validation"

        base_path = "/data/scene-rep/custom/spring/train/" # replace with your path
        split_path = "data/splits/spring/spring_split.txt"

        self.frame_paths = []
        self.flow_paths = []
        self.mask_paths = []
        files = list(glob(base_path + "*/frame_left/frame_left_*.png"))
        files = list(sorted(files))
        frameskip = cfg.model.frameskip
        with open(split_path, "r") as f:
            lines = f.readlines()
            split_map = {l[:l.index(",")]: l[l.index(",")+1:-1] for l in lines}

            for a, b, c in zip(files[:-2*frameskip],files[frameskip:-frameskip],files[2*frameskip:]):
                a_scene, b_scene, c_scene = get_scene(a, base_path), get_scene(b, base_path), get_scene(c, base_path)
                if a_scene == b_scene and b_scene == c_scene:
                    if b not in split_map:
                        raise ValueError
                    if self.split == 'all' or split_map[b] == self.split:
                        self.frame_paths.append([a, b, c])
                        self.flow_paths.append([frame_to_fwd_flow(b)])
                        self.mask_paths.append({
			    "detail": b.replace("frame_left/frame_left_", "maps/detailmap_flow_FW_left/detailmap_flow_FW_left_"),
                            "matched": b.replace("frame_left/frame_left_", "maps/matchmap_flow_FW_left/matchmap_flow_FW_left_"),
                            "rigid": b.replace("frame_left/frame_left_", "maps/rigidmap_FW_left/rigidmap_FW_left_"),
                            "sky": b.replace("frame_left/frame_left_", "maps/skymap_left/skymap_left_"),
                        })

        print("Dataset:", self.split, len(self.frame_paths))

