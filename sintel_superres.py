import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
import random

class SintelSuperResDataset():
    def __init__(self, cfg, split, resize_flow = True, dstype = None):
        self.imsz = [int(x) for x in cfg.dataset.imsz.split(",")]
        self.imsz_super = [int(x) for x in cfg.dataset.imsz_super.split(",")]
        self.split = split 
        self.idx = cfg.dataset.idx
        self.augmentations = cfg.dataset.augmentations

        assert self.split in ['training', 'validation'], "Split must be training or validation"
        assert dstype in [None, 'clean', 'final'], "Dataset type must be clean/final/None"

        self.resize_flow = resize_flow

        base_path = "/data/scene-rep/custom/Sintel/" # replace with your path
        self.path_file = "Sintel.dat"
        self.split_file = "Sintel_split.dat"

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

        self.split_paths = [] #list of frame paths [[path_to_frame1, path_to_frame2, path_to_frame3, frame2_flow]...]
        for i in range(len(pathContent)):
            if self.split == 'training' and splitContent[i][0] == '1':
                frame_num = int(pathContent[i][2])

                flow_base_path = pathContent[i][1][7:]
                flow_path = base_path + flow_base_path % frame_num

                flow_np = self.load_flow(flow_path)
                if cfg.dataset.flow_max > 0 and np.max(np.abs(flow_np)) >= cfg.dataset.flow_max:
                    continue

                png_base_path = pathContent[i][0][7:]
                frame1_path = base_path + png_base_path % (frame_num - 1)
                frame2_path = base_path + png_base_path % frame_num
                frame3_path = base_path + png_base_path % (frame_num + 1)

                self.split_paths.append([frame1_path, frame2_path, frame3_path, flow_path])
            elif self.split == 'validation' and splitContent[i][0] == '2':
                frame_num = int(pathContent[i][2])

                flow_base_path = pathContent[i][1][7:]
                flow_path = base_path + flow_base_path % frame_num

                flow_np = self.load_flow(flow_path)
                if cfg.dataset.flow_max > 0 and np.max(np.abs(flow_np)) >= cfg.dataset.flow_max:
                    continue

                png_base_path = pathContent[i][0][7:]
                frame1_path = base_path + png_base_path % (frame_num - 1)
                frame2_path = base_path + png_base_path % frame_num
                frame3_path = base_path + png_base_path % (frame_num + 1)

                self.split_paths.append([frame1_path, frame2_path, frame3_path, flow_path])
        print("Dataset:", dstype, self.split, len(self.split_paths))

        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def load_flow(self, path):
        with open(path, 'rb') as f:
            magic = float(np.fromfile(f, np.float32, count = 1)[0])
            w, h = np.fromfile(f, np.int32, count = 1)[0], np.fromfile(f, np.int32, count = 1)[0]
            data = np.fromfile(f, np.float32, count = h*w*2)
            data.resize((h, w, 2))
            return data

    def __len__(self):
        return len(self.split_paths)

    def _apply_color_transform(self, *imgs):
        jitter_params = transforms.ColorJitter.get_params(brightness=(0.6,1.4), contrast=(0.6,1.4), saturation=(0.6,1.4), hue=(-0.5/3.14,0.5/3.14))

        brightness = lambda x: transforms.functional.adjust_brightness(x, jitter_params[1])
        contrast = lambda x: transforms.functional.adjust_contrast(x, jitter_params[2])
        saturation = lambda x: transforms.functional.adjust_saturation(x, jitter_params[3])
        hue = lambda x: transforms.functional.adjust_hue(x, jitter_params[4])
        transforms_list = [brightness, contrast, saturation, hue]

        transformed_imgs = []
        for img in imgs:
            for i in jitter_params[0]:
                img = transforms_list[i](img)
            transformed_imgs.append(img)

        return transformed_imgs

    def __getitem__(self, idx):
        idx += self.idx
        paths = self.split_paths[idx]

        image1 = cv2.imread(paths[0])
        image2 = cv2.imread(paths[1])
        image3 = cv2.imread(paths[2])

        flow_np = self.load_flow(paths[3])
        if self.resize_flow:
            flow_orig = torch.tensor(flow_np).permute(2, 0, 1)
            flow_np = cv2.resize(flow_np, (self.imsz[0], self.imsz[1]))
            flow = torch.tensor(flow_np).permute(2, 0, 1)

            flow_resize_0 = 1024/self.imsz[0]
            flow_resize_1 = 436/self.imsz[1]
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
        image1 = transforms.functional.to_tensor(image1)
        image2 = transforms.functional.to_tensor(image2)
        image3 = transforms.functional.to_tensor(image3)
        image1_sup = transforms.functional.to_tensor(image1_sup)
        image2_sup = transforms.functional.to_tensor(image2_sup)
        image3_sup = transforms.functional.to_tensor(image3_sup)

        """
        # Random crop augmentation
        if self.augmentations.random_crop:
            y = int(random.random() * 800)
            x = int(random.random() * 212)
        else:
            y = 400 # half of the set values
            x = 106
        image1 = image1[:, x:x-212, y:y-800]
        image2 = image2[:, x:x-212, y:y-800]
        image3 = image3[:, x:x-212, y:y-800]
        flow = flow[:, x:x-212, y:y-800]
        image1_sup = image1_sup[:, x:x-212, y:y-800]
        image2_sup = image2_sup[:, x:x-212, y:y-800]
        image3_sup = image3_sup[:, x:x-212, y:y-800]
        """
        flow_orig = flow

        # Color augmentation
        if self.augmentations.color:
            image1_col, image1_sup_col = self._apply_color_transform(image1, image1_sup)
            image2_col, image2_sup_col = self._apply_color_transform(image2, image2_sup)
            image3_col, image3_sup_col = self._apply_color_transform(image3, image3_sup)
        else:
            image1_col, image1_sup_col = image1, image1_sup
            image2_col, image2_sup_col = image2, image2_sup
            image3_col, image3_sup_col = image3, image3_sup

        # Normalize
        image1 = self.transform(image1)
        image2 = self.transform(image2)
        image3 = self.transform(image3)
        image1_sup = self.transform(image1_sup)
        image2_sup = self.transform(image2_sup)
        image3_sup = self.transform(image3_sup)
        image1_col = self.transform(image1_col)
        image2_col = self.transform(image2_col)
        image3_col = self.transform(image3_col)
        image1_sup_col = self.transform(image1_sup_col)
        image2_sup_col = self.transform(image2_sup_col)
        image3_sup_col = self.transform(image3_sup_col)

        return image1, image2, image3, flow, image1_sup, image2_sup, image3_sup, flow_orig, image1_col, image2_col, image3_col, image1_sup_col, image2_sup_col, image3_sup_col
