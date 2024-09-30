import math
from omegaconf import OmegaConf

import torch
import torchvision.transforms as T
from PIL import Image

from featup.util import norm, unnorm, TorchPCA
from featup.plotting import plot_feats, plot_lang_heatmaps
from featup.train_implicit_upsampler import train_implicit

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

def img_to_feats(img, use_feats):
    assert(use_feats == "implicit")

    img = norm(img)
    mean = torch.Tensor([0.485, 0.456, 0.406]).to(img.device)
    std = torch.Tensor([0.229, 0.224, 0.225]).to(img.device)
    unnormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    img = unnormalize(img)

    #upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=True).to(img.device)
    #model = upsampler.model
    upsampler, model = train_implicit(torch.utils.data.DataLoader(OneImageDataset(img[0])))

    hr_feats = upsampler(img)
    """
    lr_feats = model(img)
    
    if lr_feats.shape[1] > hr_feats.shape[1]: # implicit upsampler
       lr_feats = lr_feats[:, :hr_feats.shape[1]] 
    fig = plot_feats(unnorm(img)[0], lr_feats[0], hr_feats[0])

    fig.savefig("feat_up.png")
    """
 
    return hr_feats

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
