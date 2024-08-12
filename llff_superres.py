import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from glob import glob

class LLFFSuperResDataset():
    def __init__(self, cfg, resize_flow = True):
        self.imsz = [int(x) for x in cfg.dataset.imsz.split(",")]
        self.imsz_super = [int(x) for x in cfg.dataset.imsz_super.split(",")]
        self.split = cfg.dataset.train_split
        self.idx = cfg.dataset.idx

        assert self.split in ['validation'], "Split must be validation"
        self.resize_flow = resize_flow

        base_path = f"/nobackup/nvme1/datasets/llff/{cfg.dataset.subset}/images/" # replace with your path
        #base_path = f"/home/califyn/flowmap/datasets/own/second/" # replace with your path
        
        self.split_paths = []
        files = list(glob(base_path + "*.jpg"))
        #files = list(glob(base_path + "*.png"))
        files = list(sorted(files))
        """
        for a, b, c in zip(files[:-2], files[1:-1], files[2:]):
            self.split_paths.append([a, b, c, None])
        """
        frameskip = cfg.model.frameskip
        for a, b, c in zip(files[:-2*frameskip],files[frameskip:-frameskip],files[2*frameskip:]):
            self.split_paths.append([a, b, c, None])

        print("Dataset:", self.split, len(self.split_paths))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_flow(self, path):
        with open(path, 'rb') as f:
            magic = float(np.fromfile(f, np.float32, count = 1)[0])
            w, h = np.fromfile(f, np.int32, count = 1)[0], np.fromfile(f, np.int32, count = 1)[0]
            data = np.fromfile(f, np.float32, count = h*w*2)
            data.resize((h, w, 2))
            return data

    def __len__(self):
        return len(self.split_paths)

    def __getitem__(self, idx):
        idx += self.idx
        paths = self.split_paths[idx]

        image1 = cv2.imread(paths[0])
        image2 = cv2.imread(paths[1])
        image3 = cv2.imread(paths[2])

        #flow_np = self.load_flow(paths[3])
        flow_np = np.zeros((image1.shape[0], image1.shape[1], 2)).astype('float32')
        if self.resize_flow:
            flow_orig = torch.tensor(flow_np).permute(2, 0, 1)
            flow_np = cv2.resize(flow_np, (self.imsz[0], self.imsz[1]))
            flow = torch.tensor(flow_np).permute(2, 0, 1)

            flow_resize_0 = 4032/self.imsz[0]
            flow_resize_1 = 3024/self.imsz[1]
            flow[0] = flow[0] / flow_resize_0
            flow[1] = flow[1] / flow_resize_1
        else:
            flow_orig = torch.tensor(flow_np).permute(2, 0, 1)
            flow = torch.tensor(flow_np).permute(2, 0, 1)
        
        # Convert the images from BGR to RGB
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)

        # Resize the images to a fixed size (e.g., 128x128), orignally (1024, 436, 3)
        image1_sup = cv2.resize(image1, (self.imsz_super[0], self.imsz_super[1]))
        image2_sup = cv2.resize(image2, (self.imsz_super[0], self.imsz_super[1]))
        image3_sup = cv2.resize(image3, (self.imsz_super[0], self.imsz_super[1]))

        image1 = cv2.resize(image1, (self.imsz[0], self.imsz[1]))
        image2 = cv2.resize(image2, (self.imsz[0], self.imsz[1]))
        image3 = cv2.resize(image3, (self.imsz[0], self.imsz[1]))

        # Convert the images to PyTorch tensors and normalize
        image1 = self.transform(image1)
        image2 = self.transform(image2)
        image3 = self.transform(image3)
        image1_sup = self.transform(image1_sup)
        image2_sup = self.transform(image2_sup)
        image3_sup = self.transform(image3_sup)

        """
        image1 = image1[:, 6:-7, 6:-6]
        image2 = image2[:, 6:-7, 6:-6]
        image3 = image3[:, 6:-7, 6:-6]
        flow = flow[:, 6:-7, 6:-6]
        image1_sup = image1_sup[:, 24:-28, 24:-24]
        image2_sup = image2_sup[:, 24:-28, 24:-24]
        image3_sup = image3_sup[:, 24:-28, 24:-24]
        flow_orig = flow
        image1_sup = image1_sup[:, 2:-2, :]
        image2_sup = image2_sup[:, 2:-2, :]
        image3_sup = image3_sup[:, 2:-2, :]
        """
        flow_orig = flow

        return image1, image2, image3, flow, image1_sup, image2_sup, image3_sup, flow_orig, 0, 0, 0, 0, 0, 0
