from omegaconf import OmegaConf

import torch
import torchvision.transforms as T
from PIL import Image

from featup.util import norm, unnorm
from featup.plotting import plot_feats, plot_lang_heatmaps

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

def img_to_feats(img):
    img = norm(img)

    upsampler = torch.hub.load("mhamilton723/FeatUp", 'dino16', use_norm=True).to(img.device)
    hr_feats = upsampler(img)
    lr_feats = upsampler.model(img)
    fig = plot_feats(unnorm(img)[0], lr_feats[0], hr_feats[0])

    fig.savefig("feat_up.png")

for b in dataloader:
    img_to_feats(b.frames[1])
    input("?")
