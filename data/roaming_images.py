import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
import random
import glob, math

from .superres import Batch

class RoamingImagesDataset():
    def __init__(self, cfg, split='validation'):

        self.cfg = cfg
        self.len = 300 if split == 'training' else 30

        assert(self.cfg.dataset.imsz == 1 and self.cfg.dataset.imsz_super == 1)

        self.images = glob.glob("/data/scene-rep/custom/Flickr30k/flickr30k-images/*.jpg")

        self.split = split
        assert split in ['training', 'validation'], "Split must be training or validation"
        self.crop_to = [int(x) for x in cfg.dataset.crop_to.split(",")]

    def __len__(self):
        return self.len

    def shift_subpixel(self, img, x, y):
        if x < 0 or x > 1 or y < 0 or y > 1:
            raise ValueError('shift subpixel only by values between 0 to 1')

        plus_0_mult = (1 - x) * (1 - y)
        plus_x_mult = x * (1 - y)
        plus_y_mult = y * (1 - x)
        plus_xy_mult = x * y

        plus_0_img = img
        plus_x_img = torch.cat((torch.zeros_like(img[:, :, 0, None]), img[:, :, :-1]), dim=2)
        plus_y_img = torch.cat((torch.zeros_like(img[:, 0, None, :]), img[:, :-1, :]), dim=1)
        plus_xy_img = torch.cat((torch.zeros_like(img[:, 0, None, :]), plus_x_img[:, :-1, :]), dim=1)

        return plus_0_mult * plus_0_img + plus_x_mult * plus_x_img + plus_y_mult * plus_y_img + plus_xy_mult * plus_xy_img

    def __getitem__(self, idx):
        #if self.split == "training":
        #    random.seed(idx)
        if self.split == "validation":
            random.seed(-1 * (idx + 1))
        random.shuffle(self.images)
        path_bg, path_fg = self.images[:2]
        path_bg = torchvision.io.read_image(path_bg).float() / 255.0
        path_fg = torchvision.io.read_image(path_fg).float() / 255.0

        # 256, 128
        # crop to
        fg_x, fg_y = math.floor(random.random() * (self.crop_to[0] - 100)) + 50, math.floor(random.random() * (self.crop_to[0] - 100)) + 50
        fg_w, fg_h = math.floor(random.random() * min(self.crop_to[0] - 50 - fg_x, self.crop_to[0] - 100)) + 50, math.floor(random.random() * min(self.crop_to[1] - 50 - fg_y, self.crop_to[1] - 100)) + 50 # within boundaries
        max_fl_x, max_fl_y = min(fg_x, self.crop_to[0] - fg_w - fg_x, self.cfg.dataset.flow_max), min(fg_y, self.crop_to[1] - fg_h - fg_y, self.cfg.dataset.flow_max)
        flf_x, flf_y = math.floor(random.random() * 2 * max_fl_x) - max_fl_x, math.floor(random.random() * 2 * max_fl_y) - max_fl_y
        flb_x, flb_y = math.floor(random.random() * 6) - 3, math.floor(random.random() * 6) - 3 

        path_fg = transforms.Resize((fg_w, fg_h), antialias=True)(path_fg)
        path_bg = transforms.Resize((self.crop_to[0] + 132, self.crop_to[1] + 132), antialias=True)(path_bg)
        # maybe + 66 for both?
        # 224 + 132 = 

        bg_mask = torch.ones((1, self.crop_to[0], self.crop_to[1]))
        bg_mask[:, fg_x:fg_x + fg_w, fg_y:fg_y + fg_h] = torch.zeros((fg_w, fg_h))
        fg_overlay = torch.zeros((3, self.crop_to[0], self.crop_to[1]))
        fg_overlay[:, fg_x:fg_x + fg_w, fg_y:fg_y + fg_h] = path_fg
        image2 = torch.zeros((3, self.crop_to[0], self.crop_to[1])) + bg_mask * path_bg[:, 66:self.crop_to[0] + 66, 66:self.crop_to[1] + 66] + fg_overlay

        bg_mask = torch.ones((1, self.crop_to[0], self.crop_to[1]))
        bg_mask[:, fg_x - flf_x:fg_x + fg_w - flf_x, fg_y - flf_y:fg_y + fg_h - flf_y] = torch.zeros((fg_w, fg_h))
        fg_overlay = torch.zeros((3, self.crop_to[0], self.crop_to[1]))
        fg_overlay[:, fg_x-flf_x:fg_x + fg_w - flf_x, fg_y - flf_y:fg_y + fg_h - flf_y] = path_fg
        image1 = torch.zeros((3, self.crop_to[0], self.crop_to[1])) + bg_mask * path_bg[:, 66 + flb_x:66 + self.crop_to[0] + flb_x, 66 + flb_y:66 + self.crop_to[1] + flb_y] + fg_overlay

        bg_mask = torch.ones((1, self.crop_to[0], self.crop_to[1]))
        bg_mask[:, fg_x + flf_x:fg_x + fg_w + flf_x, fg_y + flf_y:fg_y + fg_h + flf_y] = torch.zeros((fg_w, fg_h))
        fg_overlay = torch.zeros((3, self.crop_to[0], self.crop_to[1]))
        fg_overlay[:, fg_x+flf_x:fg_x + fg_w + flf_x, fg_y + flf_y:fg_y + fg_h + flf_y] = path_fg
        image3 = torch.zeros((3, self.crop_to[0], self.crop_to[1])) + bg_mask * path_bg[:, 66 - flb_x:self.crop_to[0] + 66 - flb_x, 66 - flb_y:66 + self.crop_to[1] - flb_y] + fg_overlay

        bg_mask = torch.ones((1, self.crop_to[0], self.crop_to[1]))
        bg_mask[:, fg_x:fg_x + fg_w, fg_y:fg_y + fg_h] = torch.zeros((fg_w, fg_h))
        fg_overlay = torch.zeros((2, self.crop_to[0], self.crop_to[1]))
        fg_overlay[:, fg_x:fg_x + fg_w, fg_y:fg_y + fg_h] = torch.Tensor([flf_y, flf_x]).reshape((2, 1, 1)).repeat((1, fg_w, fg_h))
        flow = torch.zeros((2, self.crop_to[0], self.crop_to[1])) + bg_mask * torch.Tensor([flb_y, flb_x]).reshape((2, 1, 1)).repeat((1, self.crop_to[0], self.crop_to[1])) + fg_overlay

        #bg_mask = torch.ones((1, self.crop_to[0], self.crop_to[1]))
        #bg_mask[:, fg_x - flf_x:fg_x + fg_w - flf_x, fg_y - flf_y:fg_y + fg_h - flf_y] = torch.zeros((fg_w, fg_h))
        #fg_overlay = torch.zeros((2, self.crop_to[0], self.crop_to[1]))
        #fg_overlay[:, fg_x - flf_x:fg_x + fg_w - flf_x, fg_y - flf_y:fg_y + fg_h - flf_y] = torch.Tensor([flf_y, flf_x]).reshape((2, 1, 1)).repeat((1, fg_w, fg_h))
        #p_flow = torch.zeros((2, self.crop_to[0], self.crop_to[1])) + bg_mask * torch.Tensor([flb_y, flb_x]).reshape((2, 1, 1)).repeat((1, self.crop_to[0], self.crop_to[1])) + fg_overlay

        # Correct dimensions (oops!)
        image1 = image1.permute(0, 2, 1)
        image2 = image2.permute(0, 2, 1)
        image3 = image3.permute(0, 2, 1)
        flow = flow.permute(0, 2, 1).flip(0)

        """
        # Aliasing
        middle = [float(x) for x in self.cfg.alias.middle.split(",")]
        image2 = self.shift_subpixel(image2, middle[0], middle[1])
        flow = self.shift_subpixel(flow, middle[0], middle[1]) - torch.Tensor(middle).to(image2.device)[:, None, None] # reduce flow if you already shifted

        end = [float(x) for x in self.cfg.alias.end.split(",")]
        image3 = self.shift_subpixel(image3, end[0], end[1])
        flow = flow + torch.Tensor(end).to(image3.device)[:, None, None]
        """

        image = [image1, image2, image3]
        return Batch(image, image, image, image, image, [flow], [flow], "blah.png", masks={})
