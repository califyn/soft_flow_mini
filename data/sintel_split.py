import matplotlib.pyplot as plt
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
import random

from superres import SuperResDataset, load_flow

base_path = "/data/scene-rep/custom/Sintel/" # replace with your path
path_file = "splits/sintel/Sintel.dat"
split_file = "splits/sintel/Sintel_split.dat"
dstype = None

pathContent_all = [i.strip().split() for i in open(path_file).readlines()]
splitContent_all = [i.strip().split() for i in open(split_file).readlines()]

pathContent = []
splitContent = []
for i in range(len(pathContent_all)):
    png_base_path = pathContent_all[i][0]
    flow_base_path = pathContent_all[i][1]
    frame_num = pathContent_all[i][2]

    pathContent.append([png_base_path, flow_base_path, frame_num])
    splitContent.append(splitContent_all[i])

flow_max_dict = {}
frame_paths = []
mask_paths = []
for i in range(len(pathContent)):
    frame_num = int(pathContent[i][2])
    
    flow_base_path = pathContent[i][1][7:]
    flow_path = base_path + flow_base_path % frame_num
    
    flow_np = load_flow(flow_path)
    
    png_base_path = pathContent[i][0][7:]
    frame2_path = base_path + png_base_path % frame_num
    flow_max_dict[frame2_path] = np.max(np.abs(flow_np))

with open("splits/sintel/sintel_max.txt", "w") as f:
    for k, v in flow_max_dict.items():
        f.write(k + "," + str(v) + "\n")
