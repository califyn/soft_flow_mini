import math
from omegaconf import OmegaConf
import os

import torch
import torchvision.transforms as T
from PIL import Image

from featup.util import norm, unnorm, TorchPCA
from featup.plotting import plot_feats, plot_lang_heatmaps
from featup.train_implicit_upsampler import train_implicit, get_common_pca

from models.unet import ResNetUNet

import wandb
from wandb_utils import download_latest_checkpoint, rewrite_checkpoint_for_compatibility
from pathlib import Path
import os
from omegaconf import OmegaConf
from src.learner import OverfitSoftLearner

"""
from data.sintel_superres import SintelSuperResDataset
from data.superres import Batch

cfg = OmegaConf.load('croco_fixed_overfit.yaml')
dataset = SintelSuperResDataset(cfg, cfg.dataset.val_split, is_val=True)
dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            collate_fn=Batch.collate_fn,
        )
"""

class OneImageDataset():
    def __init__(self, img):
        self.img = img
    
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.img

"""
unet = ResNetUNet(32)
unet.cuda()

# Load cfg and dataset
cfg = OmegaConf.load('croco_pretrain.yaml')
#dataset = SintelSuperResDataset(cfg, cfg.dataset.val_split, is_val=True)

# Set image size
#_ = dataset[0] 
cfg.dataset.imsz = cfg.dataset.crop_to
cfg.dataset.imsz_super = cfg.dataset.crop_to

# Set up logging with wandb (just using it to load model here)
with open("wandb_api_key.txt", "r") as f:
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    wandb.login(key=f.readline().strip('\n'))

# Load model
run_path = f"califyn/soft_flow/7678649213571033" # one sided run
#run_path = f"califyn/soft_flow/8095733731050450" # one sided run with minimal coloraug
print(run_path)
try:
    checkpoint_path = download_latest_checkpoint(
        run_path, Path("outputs/loaded_checkpoints")
    )
    checkpoint_path = rewrite_checkpoint_for_compatibility(checkpoint_path)
except (ValueError, wandb.errors.CommError) as e:
    print("Could not find run with run id")
    checkpoint_path = None
learner = OverfitSoftLearner.load_from_checkpoint(checkpoint_path, cfg=cfg, val_dataset=None)
learner.cuda()
learner.eval()
"""

def img_to_feats(img, use_feats, unprojector=None, save_name=None):
    cfg = OmegaConf.load("implicit_upsampler.yaml")
    feat_dim = cfg.proj_dim

    if save_name is not None:
        file_name = f'logs/feats/{save_name}_dim{feat_dim}.pt'
        if os.path.isfile(file_name):
            print("Importing feats... make sure this is what you want!")
            return torch.load(file_name)

    if use_feats == "identity":
        return img

    if use_feats == "unet":
        with torch.no_grad():
            return unet(img)

    if use_feats == "implicit":
        img = norm(img)
        mean = torch.Tensor([0.485, 0.456, 0.406]).to(img.device)
        std = torch.Tensor([0.229, 0.224, 0.225]).to(img.device)
        unnormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
        img = unnormalize(img)

        #upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=True).to(img.device)
        #model = upsampler.model
        upsampler, model = train_implicit(torch.utils.data.DataLoader(OneImageDataset(img[0])), unprojector=unprojector)

        hr_feats = upsampler(img)
        """
        lr_feats = model(img)
        
        if lr_feats.shape[1] > hr_feats.shape[1]: # implicit upsampler
           lr_feats = lr_feats[:, :hr_feats.shape[1]] 
        fig = plot_feats(unnorm(img)[0], lr_feats[0], hr_feats[0])

        fig.savefig("feat_up.png")
        """

        if save_name is not None:
            torch.save(hr_feats, f'logs/feats/{save_name}_dim{feat_dim}.pt')
     
        return hr_feats

    if use_feats == "self":
        with torch.no_grad():
            return learner.model.nn(img, img2=img)


"""
for b in dataloader:
    img_to_feats(b.frames[1].to("cuda"), 'implicit')
    input("?")
"""

def project_pca(imgs, dim):
    original_shape = list(imgs.shape)
    if imgs.ndim > 3: # to B C H W shape
         imgs = torch.reshape(imgs, (-1, *imgs.shape[-3:]))
    else:
         imgs = imgs[None]
    
    imgs = torch.permute(imgs, (0, 2, 3, 1))
    BHWC_shape = list(imgs.shape)
    imgs = torch.reshape(imgs, (-1, imgs.shape[-1])) # to M N shape

    pca = TorchPCA(dim)
    pca.fit(imgs)
    proj = pca.transform(imgs)

    BHWC_shape[-1] = dim
    proj = torch.reshape(proj, BHWC_shape)
    proj = torch.permute(proj, (0, 3, 1, 2))
    
    original_shape[-3] = dim
    return torch.reshape(proj, original_shape)
