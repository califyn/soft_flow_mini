import torch
import math
import numpy as np

import wandb
from wandb_utils import download_latest_checkpoint, rewrite_checkpoint_for_compatibility
from pathlib import Path
import os
from tqdm import tqdm
from copy import deepcopy

from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import PIL

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600

from src.learner import OverfitSoftLearner
from src.soft_utils import tiled_pred
from data.sintel_superres import SintelSuperResDataset
from data.superres import Batch

# Load cfg and dataset
cfg = OmegaConf.load('croco_fixed_overfit.yaml')
dataset = SintelSuperResDataset(cfg, cfg.dataset.val_split, is_val=True)

# Set image size
_ = dataset[0] 
cfg.dataset.imsz = cfg.dataset.crop_to
cfg.dataset.imsz_super = cfg.dataset.crop_to

# Set up logging with wandb (just using it to load model here)
with open("wandb_api_key.txt", "r") as f:
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    wandb.login(key=f.readline().strip('\n'))

# Load model
#run_path = f"califyn/soft_flow/7678649213571033" # one sided run
#run_path = f"califyn/soft_flow/8095733731050450" # one sided run with minimal coloraug
#run_path = f"califyn/soft_flow/8097297764129804" # one sided run with maximal coloraug
run_path = f"calify/soft_flow/8367374615662478" # strong masking
print(run_path)
try:
    checkpoint_path = download_latest_checkpoint(
        run_path, Path("outputs/loaded_checkpoints")
    )
    checkpoint_path = rewrite_checkpoint_for_compatibility(checkpoint_path)
except (ValueError, wandb.errors.CommError) as e:
    print("Could not find run with run id")
    checkpoint_path = None
#learner = OverfitSoftLearner(cfg, val_dataset=dataset)
learner = OverfitSoftLearner.load_from_checkpoint(checkpoint_path, cfg=cfg, val_dataset=dataset)
learner.eval()

# Fix dataset
dataset.frame_paths = [dataset.frame_paths[cfg.dataset.idx]]
dataset.flow_paths = [dataset.flow_paths[cfg.dataset.idx]]
dataset.mask_paths = [dataset.mask_paths[cfg.dataset.idx]]
dataset.idx = 0
dataset.return_uncropped = True

# Dataloader
loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            collate_fn=Batch.collate_fn
        )

# Gets features with tiled pred
def get_feats_fwd(b):
    b.to(batch.frames[0].device)
    t_patch = math.ceil((learner.model.weight_sl//2)/16)*16

    def get_dprod(bb):
        input_2 = torch.nn.functional.pad(bb.frames[2], (t_patch, t_patch, t_patch, t_patch), mode="constant")
        input_1 = torch.nn.functional.pad(bb.frames[1], (t_patch, t_patch, t_patch, t_patch), mode="constant")
        with torch.no_grad():
            input_2 = input_2.cuda()
            input_1 = input_1.cuda()
            src_feats = learner.model.nn(input_2, input_1)
            tgt_feats = learner.model.nn(input_1, input_2)
            src_feats = src_feats.cpu()
            tgt_feats = tgt_feats.cpu()
        src_feats = src_feats / torch.linalg.norm(src_feats, dim=1, keepdim=True)
        tgt_feats = tgt_feats / torch.linalg.norm(tgt_feats, dim=1, keepdim=True)

        return torch.cat([src_feats[:, :, t_patch:-t_patch, t_patch:-t_patch], tgt_feats[:, :, t_patch:-t_patch, t_patch:-t_patch]], dim=1)

    occ = tiled_pred(get_dprod, b, cfg.dataset.flow_max, lambda x, y: dataset.crop_batch(x, crop=y), crop=(224, 224), temp=learner.temp, out_key='given', loss_fn=learner.loss, given_dim=learner.cfg.model.feat_dim * 2, do_att_map=False)
    return occ

# Get features from first batch
for batch in loader:
    tgt_frame = batch.frames_orig[1]
    src_frame = batch.frames_orig[2]
    flow = batch.flow_orig[0]

    feats = get_feats_fwd(batch)
    src_feats = feats[:, :learner.cfg.model.feat_dim]
    tgt_feats = feats[:, learner.cfg.model.feat_dim:]

# Compute the attention map with query idx
def get_att_map(query_idx, softmax=False):
    query = tgt_feats[..., query_idx[0], query_idx[1]]

    att_map = torch.sum(src_feats * query[:, :, None, None], dim=1, keepdim=True)
    assert(att_map.shape[0] == 1)

    if softmax:
        att_map_shape = att_map.shape
        att_map = att_map.reshape((att_map.shape[0], att_map.shape[1], -1))
        att_map = torch.nn.functional.softmax(att_map * learner.temp.cpu() / 2, dim=-1)
        att_map = att_map.reshape(att_map_shape)

        return att_map[0].repeat(3, 1, 1)
    else:
        fig = plt.figure(frameon=False)
        fig.set_size_inches(10.24, 4.36)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax.imshow(att_map[0, 0], cmap='RdBu_r', vmin=-1., vmax=1., aspect='equal')

        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        data = torch.Tensor(data)
        data = torch.permute(data, (2, 0, 1))
        return data / 255.

# Add lines so we know where things are
def impose_lines(img, query_idx, center=True):
    assert(img.shape[0] == 3 and img.ndim == 3)
    img = deepcopy(torch.Tensor(img.detach().numpy()))
    if center:
        img[0, query_idx[0], :] = img[0, query_idx[0], :] * 0.1
        img[1, query_idx[0], :] = 1 - (1 - img[1, query_idx[0], :]) * 0.1
        img[0, :, query_idx[1]] = img[0, :, query_idx[1]] * 0.1
        img[2, :, query_idx[1]] = 1 - (1 - img[2, :, query_idx[1]]) * 0.1
    else:
        img[0, query_idx[0]-2, :] = img[0, query_idx[0]-2, :] * 0.1
        img[1, query_idx[0]-2, :] = 1 - (1 - img[1, query_idx[0]-2, :]) * 0.1
        img[0, query_idx[0]+2, :] = img[0, query_idx[0]+2, :] * 0.1
        img[1, query_idx[0]+2, :] = 1 - (1 - img[1, query_idx[0]+2, :]) * 0.1
        img[0, :, query_idx[1]-2] = img[0, :, query_idx[1]-2] * 0.1
        img[2, :, query_idx[1]-2] = 1 - (1 - img[2, :, query_idx[1]-2]) * 0.1
        img[0, :, query_idx[1]+2] = img[0, :, query_idx[1]+2] * 0.1
        img[2, :, query_idx[1]+2] = 1 - (1 - img[2, :, query_idx[1]+2]) * 0.1

    return img

# Get visualization for a given query_idx
def get_vis(query_idx):
    frame1 = impose_lines(tgt_frame[0], query_idx)

    warped_query = list(torch.round(torch.Tensor(query_idx) + torch.flip(batch.flow_orig[0][0, :, query_idx[0], query_idx[1]], dims=[0])).int())
    frame2 = impose_lines(src_frame[0], warped_query)
    att = get_att_map(query_idx)
    att = torch.nn.functional.interpolate(att[None], size=frame2.shape[-2:], mode='bilinear')[0]
    att = impose_lines(att, warped_query, center=False)
    att_softmax = get_att_map(query_idx, softmax=True)
    att_softmax = impose_lines(att_softmax, warped_query, center=False)

    img = torch.cat((frame1, frame2, att, att_softmax, torch.clamp(100 * att_softmax, min=0, max=1)), dim=-2)
    img = torch.permute(img, (1, 2, 0))

    fig = plt.figure(frameon=False)
    fig.set_size_inches(2.5, 8)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img.detach().cpu(), aspect='equal')
    plt.savefig('att.png')
    plt.close()

import imageio
images = []
x_list = np.linspace(0.1 * src_frame.shape[2], 0.9 * src_frame.shape[2], num=1000)
y_list = np.linspace(0.1 * src_frame.shape[3], 0.9 * src_frame.shape[3], num=1000)
for x, y in tqdm(zip(x_list, y_list)):
    get_vis([int(x), int(y)])
    image = imageio.imread('att.png')
    images.append(image)

imageio.mimsave("att.mp4", images, format='MP4')
"""
with imageio.get_writer('./att.gif', mode='I') as writer:
    x_list = np.linspace(0.1 * src_frame.shape[2], 0.9 * src_frame.shape[2], num=20)
    y_list = np.linspace(0.1 * src_frame.shape[3], 0.9 * src_frame.shape[3], num=20)
    for x, y in tqdm(zip(x_list, y_list)):
        get_vis([int(x), int(y)])
        image = imageio.imread('att.png')
        writer.append_data(image)
"""
