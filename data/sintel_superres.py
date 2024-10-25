import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
import random
import os

from .superres import SuperResDataset, load_flow

class SintelSuperResDataset(SuperResDataset):
    def load_paths(self, cfg, split):
        assert self.split in ['training', 'validation', 'all'], "Split must be training or validation"
        if not hasattr(cfg.dataset, 'skip_forward') or cfg.dataset.skip_forward is None:
            cfg.dataset.skip_forward = 1

        base_path = "/data/scene-rep/custom/Sintel/" # replace with your path
        self.path_file = "data/splits/sintel/Sintel.dat"
        self.split_file = "data/splits/sintel/Sintel_split.dat"
        dstype = cfg.dataset.dstype

        pathContent_all = [i.strip().split() for i in open(self.path_file).readlines()]
        splitContent_all = [i.strip().split() for i in open(self.split_file).readlines()]

        pathContent = []
        splitContent = []
        for i in range(len(pathContent_all)):
            png_base_path = pathContent_all[i][0]
            flow_base_path = pathContent_all[i][1]
            frame_num = pathContent_all[i][2]

            if dstype == None:
                pathContent.append([png_base_path, flow_base_path, frame_num])
                splitContent.append(splitContent_all[i])
            elif dstype == 'clean' and 'clean' in png_base_path:
                pathContent.append([png_base_path, flow_base_path, frame_num])
                splitContent.append(splitContent_all[i])
            elif  dstype == 'final' and 'final' in png_base_path:
                pathContent.append([png_base_path, flow_base_path, frame_num])
                splitContent.append(splitContent_all[i])
        flow_max = {}
        with open("data/splits/sintel/sintel_max.txt", "r") as f:
            for l in f.readlines():
                flow_max[l[:l.index(",")]] = float(l[l.index(",")+1:])

        self.frame_paths = []
        self.flow_paths = []
        self.mask_paths = []
        for i in range(len(pathContent)):
            if (self.split == 'training' and splitContent[i][0] == '1') or self.split == 'all':
                frame_num = int(pathContent[i][2])

                flow_base_path = pathContent[i][1][7:]
                flow_path = base_path + flow_base_path % frame_num

                flow_np = load_flow(flow_path)

                png_base_path = pathContent[i][0][7:]
                frame1_path = base_path + png_base_path % (frame_num - 1)
                frame2_path = base_path + png_base_path % frame_num
                frame3_path = base_path + png_base_path % (frame_num + cfg.dataset.skip_forward)
                if cfg.dataset.flow_max > 0 and flow_max[frame2_path] >= cfg.dataset.flow_max:
                    continue

                if os.path.exists(frame3_path):
                    self.frame_paths.append([frame1_path, frame2_path, frame3_path])
                    self.flow_paths.append([flow_path])
                    self.mask_paths.append({
                        "occ": frame2_path.replace(dstype, "occlusions")
                    })
            elif (self.split == 'validation' and splitContent[i][0] == '2') or self.split == 'all':
                frame_num = int(pathContent[i][2])

                flow_base_path = pathContent[i][1][7:]
                flow_path = base_path + flow_base_path % frame_num

                flow_np = load_flow(flow_path)

                png_base_path = pathContent[i][0][7:]
                frame1_path = base_path + png_base_path % (frame_num - 1)
                frame2_path = base_path + png_base_path % frame_num
                frame3_path = base_path + png_base_path % (frame_num + cfg.dataset.skip_forward)
                if cfg.dataset.flow_max > 0 and flow_max[frame2_path] >= cfg.dataset.flow_max:
                    continue

                if os.path.exists(frame3_path):
                    self.frame_paths.append([frame1_path, frame2_path, frame3_path])
                    self.flow_paths.append([flow_path])
                    self.mask_paths.append({
                        "occ": frame2_path.replace(dstype, "occlusions")
                    })
        print("Sintel Dataset:", dstype, self.split, len(self.frame_paths))
