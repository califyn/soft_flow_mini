import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from copy import deepcopy
import random
import h5py

from src.feats import img_to_feats, project_pca

class Batch():
    def __init__(self, frames, frames_up, frames_orig, frames_col, frames_up_col, frames_feat, frames_up_feat, flow, flow_orig, path, masks={}, collate_different_shapes="error", crop_masks={}):
        self.frames = frames
        self.frames_up = frames_up
        self.frames_orig = frames_orig
        self.frames_col = frames_col
        self.frames_up_col = frames_up_col
        self.frames_feat = frames_feat
        self.frames_up_feat = frames_up_feat
        self.flow = flow
        self.flow_orig = flow_orig
        self.path = path 
        self.masks = masks
        self.collate_different_shapes = collate_different_shapes
        self.crop_masks = crop_masks

        """
        print({
            "frames": [x.mean() for x in self.frames],
            "frames_up": [x.mean() for x in self.frames_up],
            "frames_col": [x.mean() for x in self.frames_col],
            "frames_up_col": [x.mean() for x in self.frames_up_col],
            "flow": None if self.flow is None else [x.mean() for x in self.flow],
            "flow_orig": None if self.flow_orig is None else [x.mean() for x in self.flow_orig],
        })
        input("?")
        """

    def to(self, device):
        self.frames = [x.to(device) for x in self.frames]
        self.frames_up = [x.to(device) for x in self.frames_up]
        self.frames_orig = [x.to(device) for x in self.frames_orig]
        self.frames_col = [x.to(device) for x in self.frames_col]
        self.frames_up_col = [x.to(device) for x in self.frames_up_col]
        self.frames_feat = [x.to(device) for x in self.frames_feat]
        self.frames_up_feat = [x.to(device) for x in self.frames_up_feat]
        if self.flow is not None:
            self.flow = [x.to(device) for x in self.flow]
        if self.flow_orig is not None:
            self.flow_orig = [x.to(device) for x in self.flow_orig]
        for name, mask in self.masks.items():
            self.masks[name] = self.masks[name].to(device)

    @staticmethod
    def collate_fn(list_instances): # does NOT support mixing datasets
        assert(list_instances[0].frames[0].ndim == 3)
        
        """
        shapes = [tuple(list(b.frames[0][:2])) for b in list_instances]
        all_equal_shapes = all([s == shapes[0] for s in shapes])
        if not all_equal_shapes:
            handler = list_instances[0].collate_different_shapes
            if handler == "error":
                raise ValueError("attempted to collate different shapes")
            elif handler == "first":
                list_instances = [l for l, s in zip(list_instances, shapes) if s == shapes[0]]

        """
        return Batch(
            [torch.stack([b.frames[i] for b in list_instances], dim=0) for i in range(len(list_instances[0].frames))],
            [torch.stack([b.frames_up[i] for b in list_instances], dim=0) for i in range(len(list_instances[0].frames_up))],
            [torch.stack([b.frames_orig[i] for b in list_instances], dim=0) for i in range(len(list_instances[0].frames_orig))],
            [torch.stack([b.frames_col[i] for b in list_instances], dim=0) for i in range(len(list_instances[0].frames_col))],
            [torch.stack([b.frames_up_col[i] for b in list_instances], dim=0) for i in range(len(list_instances[0].frames_up_col))],
            [torch.stack([b.frames_feat[i] for b in list_instances], dim=0) for i in range(len(list_instances[0].frames_feat))],
            [torch.stack([b.frames_up_feat[i] for b in list_instances], dim=0) for i in range(len(list_instances[0].frames_up_feat))],
            None if list_instances[0].flow is None else [torch.stack([b.flow[i] for b in list_instances], dim=0) for i in range(len(list_instances[0].flow))],
            None if list_instances[0].flow is None else [torch.stack([b.flow_orig[i] for b in list_instances], dim=0) for i in range(len(list_instances[0].flow_orig))],
            [b.path for b in list_instances],
            masks={name: torch.stack([b.masks[name] for b in list_instances], dim=0) for name in list_instances[0].masks.keys()},
            crop_masks={name: torch.stack([b.crop_masks[name] for b in list_instances], dim=0) for name in list_instances[0].crop_masks.keys()},
        )

    @staticmethod
    def index(batch, idx): # Index along the FRAME dimension
        assert(batch.frames[0].ndim == 4)
        if isinstance(idx, list):
            return Batch(
               [batch.frames[i] for i in idx],
               [batch.frames_up[i] for i in idx],
               [batch.frames_orig[i] for i in idx],
               [batch.frames_col[i] for i in idx],
               [batch.frames_up_col[i] for i in idx],
               [batch.frames_feat[i] for i in idx],
               [batch.frames_up_feat[i] for i in idx],
               batch.flow,
               batch.flow_orig,
               batch.path,
               masks=batch.masks,
               crop_masks=batch.crop_masks,
            )
        else:
            raise ValueError('idx for batch should be list')

def load_flow(path):
    if path.endswith(".flo"):
        with open(path, 'rb') as f:
            magic = float(np.fromfile(f, np.float32, count = 1)[0])
            w, h = np.fromfile(f, np.int32, count = 1)[0], np.fromfile(f, np.int32, count = 1)[0]
            data = np.fromfile(f, np.float32, count = h*w*2)
            data.resize((h, w, 2))
            return data
    elif path.endswith(".npy"):
        return np.load(path)
    elif path.endswith(".flo5"):
        with h5py.File(path, "r") as f:
            if "flow" not in f.keys():
                raise IOError(f"File {filename} does not have a 'flow' key. Is this a valid flo5 file?")
            return f["flow"][()]
    elif path.endswith(".png"): # KITTI
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = img.astype('f')
        nan_mask = img[..., 0, None]

        img = np.flip(img[..., 1:], axis=-1)
        img = (img - 2**15) / 64

        img = np.where(nan_mask, img, np.full_like(img, float('nan')))
        return img
    else:
        raise ValueError(f"{path} doesn't have extension flo or npy")

class SuperResDataset():
    def __init__(self, cfg, split, is_val=False):
        if not isinstance(cfg.dataset.imsz, int):
            self.imsz = [int(x) for x in cfg.dataset.imsz.split(",")]
        else:
            self.imsz = None
            self.imsz_res = int(cfg.dataset.imsz)
        if not isinstance(cfg.dataset.imsz_super, int):
            self.imsz_super = [int(x) for x in cfg.dataset.imsz_super.split(",")]
        else:
            self.imsz_super = None
            self.imsz_super_res = int(cfg.dataset.imsz_super)
        self.split = split 
        self.idx = cfg.dataset.idx
        self.augmentations = cfg.dataset.augmentations
        self.crop_to = None if cfg.dataset.crop_to is None else [int(x) for x in cfg.dataset.crop_to.split(",")]
        self.use_feats = cfg.dataset.use_feats
        self.img_as_feat = None

        self.load_paths(cfg, split)
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.flow_multiplier = 1 # 2 for spring, 1 otherwise
        self.collate_different_shapes_mode = "error"
        self.iters = 0 # for validation randomness
        self.return_uncropped = False
        self.is_val = is_val

        self.is_overfitting = cfg.training.overfit_batch == 1

    def load_paths(self, *args, **kwargs):
        raise NotImplementedError("need specific dataset!")

    def __len__(self):
        return len(self.frame_paths)

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

    def set_imsz(self, input_imsz):
        self.imsz_super_orig = self.imsz_super
        max_res_ratio = max(self.imsz_res, self.imsz_super_res)
        #if input_imsz[0] % max_res_ratio != 0 or input_imsz[1] % max_res_ratio != 0:
        x_excess, y_excess = input_imsz[0] % max_res_ratio, input_imsz[1] % max_res_ratio
        self.image_trim = (x_excess // 2, (x_excess + 1) // 2, y_excess // 2, (y_excess + 1) // 2)

        self.imsz = (input_imsz[0] // self.imsz_res, input_imsz[1] // self.imsz_res)
        #self.imsz_super = (input_imsz[0] // self.imsz_super_res, input_imsz[1] // self.imsz_super_res)
        assert(self.imsz_res % self.imsz_super_res == 0)
        ratio = self.imsz_res // self.imsz_super_res
        self.imsz_super = (self.imsz[0] * ratio, self.imsz[1] * ratio)
        print(f"Set image size to {self.imsz} and image UP size to {self.imsz_super}")

    def np_crop(self, img, image_trim):
        img = img[image_trim[2]:, image_trim[0]:]
        if image_trim[3] > 0:
            img = img[:-image_trim[3], :]
        if image_trim[1] > 0:
            img = img[:, :-image_trim[1]]

        return img

    def crop(self, img, image_trim):
        if image_trim[1] > 0:
            img = img[..., :-image_trim[1], :]
        if image_trim[3] > 0:
            img = img[..., :, :-image_trim[3]]
        img = img[..., image_trim[0]:, image_trim[2]:]
        """

        if image_trim[0] < 0:
            img = torch.nn.functional.pad(img, (0, 0, -image_trim[0], 0), mode="constant")
        else image_trim[0] > 0:
            img = img[:, image_trim[0]:, :]

        if image_trim[1] < 0:
            img = torch.nn.functional.pad(img, (0, 0, 0, -image_trim[1]), mode="constant")
        else image_trim[1] > 0:
            img = img[:, :-image_trim[1], :]

        if image_trim[2] < 0:
            img = torch.nn.functional.pad(img, (-image_trim[2], 0, 0, 0), mode="constant")
        else image_trim[2] > 0:
            img = img[:, :, image_trim[2]:]

        if image_trim[3] < 0:
            img = torch.nn.functional.pad(img, (0, -image_trim[3], 0, 0), mode="constant")
        else image_trim[3] > 0:
            img = img[:, :, :-image_trim[3]]
 
        """
        return img

    def gen_flow_masks(self, flow_orig, shrink=False):
        assert(len(flow_orig) == 1)
        flow_mags = torch.linalg.norm(flow_orig[0], dim=0, keepdim=True)

        return {
            "s0-10": flow_mags < 10,
            "s10-40": torch.logical_and(flow_mags >= 10.0, flow_mags < 40.0),
            "s40+": flow_mags >= 40.0
        }

    def gen_inverse_masks(self, masks):
        not_masks = {"not_" + name: 1-mask for name, mask in masks.items()}

        return masks | not_masks

    def crop_batch(self, batch, crop=None, idx=None):
        # ONLY APPLY THIS ONCE ! Flow orig will get messed up applying this multiple times
        batch = deepcopy(batch)

        if crop is None:
            x_extra = batch.frames[0].shape[1] - self.crop_to[0]
            y_extra = batch.frames[0].shape[2] - self.crop_to[1]
            if self.augmentations.random_crop:
                if self.is_val:
                    #random.seed(1000 * (self.iters % 4) + idx)
                    random.seed(1000 * 1 + idx)
                x1 = int(random.random() * x_extra)
                y1 = int(random.random() * y_extra)
            else:
                x1 = (x_extra + 1) // 2
                y1 = (y_extra + 1) // 2
            x2 = x_extra - x1
            y2 = y_extra - y1
        else:
           x1, x2, y1, y2 = crop
        
        up_ratio = batch.frames_up[0].shape[2] // batch.frames[0].shape[2]
        for i, p in enumerate(batch.frames):
            batch.frames[i] = self.crop(p, [x1, x2, y1, y2])
        for i, p in enumerate(batch.frames_up):
            batch.frames_up[i] = self.crop(p, [x1 * up_ratio, x2 * up_ratio, y1 * up_ratio, y2 * up_ratio])
        for i, p in enumerate(batch.frames_col):
            batch.frames_col[i] = self.crop(p, [x1, x2, y1, y2])
        for i, p in enumerate(batch.frames_up_col):
            batch.frames_up_col[i] = self.crop(p, [x1 * up_ratio, x2 * up_ratio, y1 * up_ratio, y2 * up_ratio])
        for i, p in enumerate(batch.flow):
            batch.flow[i] = self.crop(p, [x1, x2, y1, y2])
        up_ratio *= self.imsz_super_res
        for i, p in enumerate(batch.flow_orig):
            batch.flow_orig[i] = self.crop(p, [x1 * up_ratio + self.image_trim[2], x2 * up_ratio + self.image_trim[3], y1 * up_ratio + self.image_trim[0], y2 * up_ratio + self.image_trim[1]])

        batch.crop_masks = {}
        for k, v in batch.masks.items():
            assert(up_ratio == 1)
            batch.crop_masks[k] = self.crop(v, [x1, x2, y1, y2])
        return batch

    def __getitem__(self, idx):
        if idx == 0:
            self.iters += 1
        idx += self.idx
        frame_paths = self.frame_paths[idx]
        if self.flow_paths is not None:
            flow_paths = self.flow_paths[idx]
        else:
            flow_paths = None
        mask_paths = self.mask_paths[idx]

        image = []
        for p in frame_paths:
            image.append(cv2.imread(p))

        masks = {}
        for name, p in mask_paths.items():
            masks[name] = cv2.imread(p)
            masks[name] = torch.Tensor(masks[name])[None, ..., 0]
            if masks[name].shape[1] == 2 * image[0].shape[0] and masks[name].shape[2] == 2 * image[0].shape[1]:
                masks[name] = masks[name][:, ::2, ::2]
            masks[name][masks[name]==255] = 1

        if self.imsz is None:
            self.set_imsz((image[0].shape[1], image[0].shape[0]))
        for i, p in enumerate(image):
            image[i] = self.np_crop(p, self.image_trim)

        if flow_paths is not None:
            flow = []
            flow_orig = []
            for p in flow_paths:
                next_flow = load_flow(p).astype(np.float32) * self.flow_multiplier
                flow.append(next_flow)
                flow_orig.append(torch.tensor(flow[-1]).permute(2, 0, 1))
                orig_size = (flow_orig[-1].shape[-2], flow_orig[-1].shape[-1])

                flow[-1] = self.np_crop(flow[-1], self.image_trim)
                flow[-1] = cv2.resize(flow[-1], (self.imsz[0], self.imsz[1]))
                flow[-1] = torch.tensor(flow[-1]).permute(2, 0, 1)
                flow[-1][0] = flow[-1][0] * (self.imsz[0]/orig_size[1])
                flow[-1][1] = flow[-1][1] * (self.imsz[1]/orig_size[0])
        else:
            flow = None
            flow_orig = None
        
        # Convert the images from BGR to RGB
        for i, p in enumerate(image):
            image[i] = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)

        image_orig = []
        for p in image:
            image_orig.append(p)

        # Resize the images to a fixed size (e.g., 128x128), orignally (1024, 436, 3)
        image_up = []
        for p in image:
            image_up.append(cv2.resize(p, (self.imsz_super[0], self.imsz_super[1])))

        for i, p in enumerate(image):
            image[i] = cv2.resize(p, (self.imsz[0], self.imsz[1]))

        # Convert the images to PyTorch tensors and normalize
        for i, p in enumerate(image):
            image[i] = transforms.functional.to_tensor(p)
        for i, p in enumerate(image_orig):
            image_orig[i] = transforms.functional.to_tensor(p)
        for i, p in enumerate(image_up):
            image_up[i] = transforms.functional.to_tensor(p)

        # Color augmentation
        image_col, image_up_col = [], []
        if self.augmentations.color:
            for p, q in zip(image, image_up):
                p_transformed, q_transformed = self._apply_color_transform(p, q)
                image_col.append(p_transformed)
                image_up_col.append(q_transformed)
        else:
            image_col, image_up_col = deepcopy(image), deepcopy(image_up)

        # Normalize
        for i, p in enumerate(image):
            image[i] = self.transform(p)
        for i, p in enumerate(image_up):
            image_up[i] = self.transform(p)
        for i, p in enumerate(image_col):
            image_col[i] = self.transform(p)
        for i, p in enumerate(image_up_col):
            image_up_col[i] = self.transform(p)

        masks = self.gen_inverse_masks(masks)
        flow_masks = self.gen_flow_masks(flow_orig)
        if flow_orig[0].shape[1] == 2 * image_orig[0].shape[1] and flow_orig[0].shape[2] == 2 * image_orig[0].shape[2]:
            flow_masks = {k: v[:, ::2, ::2] for k, v in flow_masks.items()}
        masks = masks | flow_masks
        
        batch = Batch(image, image_up, image_orig, image_col, image_up_col, None, None, flow, flow_orig, frame_paths[1], masks=masks, collate_different_shapes=self.collate_different_shapes_mode)
        if not self.return_uncropped and self.crop_to is not None:
            batch = self.crop_batch(batch, idx=idx)

        if self.use_feats is not None:
            # Not handling downsampling for now
            assert(batch.frames[i].shape[-1] == batch.frames_up[i].shape[-1])
            assert(self.is_overfitting)
            assert(not self.augmentations.random_crop)
            if self.img_as_feat is None:
                self.img_as_feat = []
                batch.frames_feat = [None] * len(batch.frames)
                batch.frames_up_feat = [None] * len(batch.frames_up)
                for i, p in enumerate(batch.frames):
                    img_as_feat = img_to_feats(p[None].to("cuda"), self.use_feats)[0].cpu()
                    self.img_as_feat.append(torch.Tensor(img_as_feat.detach().cpu().numpy()))
                projected = project_pca(torch.stack(self.img_as_feat, dim=0), 64)
                assert(self.img_as_feat[0].shape)
                self.img_as_feat = [x[0] for x in torch.chunk(projected, 3, dim=0)]
            batch.frames_feat = deepcopy(self.img_as_feat)
            batch.frames_up_feat = deepcopy(self.img_as_feat)
        else:
            batch.frames_feat = deepcopy(batch.frames)
            batch.frames_up_feat = deepcopy(batch.frames_up)
         
        return batch
