import lightning as L
from omegaconf import DictConfig
import math
import functools
from copy import deepcopy
import time

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import cv2

from torchvision.models.optical_flow import raft_large
from smurf import raft_smurf

from .soft_losses import *
from .soft_utils import *
from .softsplat_downsample import softsplat
from src import eigenutils
from .unet import HalfUnet, Unet

from models.croco import CroCoNet
from models.croco_downstream import croco_args_from_ckpt, CroCoDownstreamBinocular
from models.head_downstream import PixelwiseTaskWithDPT
from models.pos_embed import interpolate_pos_embed
from .model_utils import get_parameter_groups

from PIL import Image
import torchvision.transforms
from torchvision.transforms import ToTensor, Normalize, Compose

import wandb

def patchify(p, imgs):
    """
    imgs: (B, 3, H, W)
    x: (B, L, patch_size**2 *3)
    """
    assert imgs.shape[2] % p == 0
    assert imgs.shape[3] % p == 0

    h = imgs.shape[2] // p
    w = imgs.shape[3] // p
    x = imgs.reshape(shape=(imgs.shape[0], -1, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, -1))
    
    return x

def unpatchify(patch_size, h, w, x):
    """
    x: (N, L, patch_size**2 *channels)
    imgs: (N, 3, H, W)
    """
    if h is None and w is None:
        h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]
    x = x.reshape(shape=(x.shape[0], h, w, patch_size, patch_size, -1))
    channels = x.shape[-1]
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], channels, h * patch_size, w * patch_size))
    return imgs


class CroCoWrapper(torch.nn.Module):
    def __init__(self, feat_dim,):
        super(CroCoWrapper, self).__init__()
        ckpt = torch.load('./CroCo_V2_ViTBase_SmallDecoder.pth', 'cpu')
        model = CroCoNet( **ckpt.get('croco_kwargs',{}))
        msg = model.load_state_dict(ckpt['model'], strict=True)
    
        self.model = model
        self.feat_dim = feat_dim

        old_prediction_head = self.model.prediction_head
        self.model.prediction_head = torch.nn.Linear(512, 256*feat_dim)
        
        old_weight = old_prediction_head._parameters['weight'].data
        old_bias = old_prediction_head._parameters['bias'].data
        old_weight = old_weight.reshape((16, 16, 3, 512))
        old_bias = old_bias.reshape((16, 16, 3))
        old_weight = old_weight.repeat(1, 1, feat_dim//3 + 1, 1)[..., :feat_dim, :]
        old_bias = old_bias.repeat(1, 1, feat_dim//3 + 1)[..., :feat_dim]
        old_weight = old_weight.reshape((-1, 512))
        old_bias = old_bias.reshape((-1, ))

        self.model.prediction_head._parameters['weight'].data = old_weight 
        self.model.prediction_head._parameters['bias'].data = old_bias

    def forward(self, img1, img2=None):
        self.model.patch_embed.img_size = img1.shape[-2:]

        #out, _, _ = self.model(img1, img2)
        out, _, _ = self.model(img1, img1)
        return self.model.unpatchify(out, channels=self.feat_dim, h=img1.shape[-2]//16, w=img1.shape[-1]//16)

class CroCoDPTWrapper(torch.nn.Module):
    def __init__(self, feat_dim, both_sided=True):
        super(CroCoDPTWrapper, self).__init__()
        ckpt = torch.load('./CroCo_V2_ViTBase_SmallDecoder.pth', 'cpu')
        croco_args = croco_args_from_ckpt(ckpt)

        self.head = PixelwiseTaskWithDPT()
        self.head.num_channels = feat_dim
        model = CroCoDownstreamBinocular(self.head, **croco_args)
        interpolate_pos_embed(model, ckpt['model'])
        msg = model.load_state_dict(ckpt['model'], strict=False)
    
        self.model = model
        self.feat_dim = feat_dim
        self.both_sided = both_sided

    def forward(self, img1, img2=None):
        self.model.patch_embed.img_size = img1.shape[-2:]

        if self.both_sided:
            return self.model(img1, img2)
        else:
            return self.model(img1, img1)

class SoftMultiFramePredictor(torch.nn.Module):
    def __init__(self, cfg, n_frames, weight_sl=31, downsample_factor=1, filter_zoom=1, init='all', feat_dim=256):
        super(SoftMultiFramePredictor, self).__init__()
        image_size = [int(x) for x in cfg.dataset.imsz.split(",")]
        assert((weight_sl + downsample_factor - 1) % filter_zoom == 0 and weight_sl % 2 == 1)

        self.weight_sl = weight_sl
        self.downsample_factor = downsample_factor
        self.filter_zoom = filter_zoom
        self.true_sl = (weight_sl + downsample_factor - 1) // filter_zoom

        if cfg.model.border_handling_on >= 0:
            self.border_handling = None
        else:
            self.border_handling = cfg.model.border_handling

        t = self.weight_sl//2 + (self.downsample_factor - 1)/2
        x_positions = torch.linspace(-t, t, self.weight_sl + self.downsample_factor - 1)
        y_positions = torch.linspace(-t, t, self.weight_sl + self.downsample_factor - 1)
        grid_x, grid_y = torch.meshgrid(x_positions, y_positions)
        self.grid = torch.stack((grid_x, grid_y), dim=-1) 
        self.grid = torch.reshape(self.grid, (self.true_sl, self.filter_zoom, self.true_sl, self.filter_zoom, -1))
        self.grid = torch.mean(self.grid, dim=(1, 3))

        self.max_size = torch.max(torch.abs(self.grid)) / self.downsample_factor

        self.method = cfg.cost.method
        self.pred_occ_mask = cfg.cost.pred_occ_mask

        H, W = image_size[1], image_size[0]
        if self.method == 'weights':
            self.weights = torch.nn.Parameter(torch.zeros(1, n_frames-1, H, W, self.true_sl, self.true_sl)) 
        elif self.method == 'feats':
            fill_val = (1/feat_dim) ** 0.5
            self.feats = torch.nn.Parameter(torch.full((1, n_frames, feat_dim, H, W), fill_val))
        elif self.method == 'nn':
            if cfg.cost.model == "croco":
                self.nn = CroCoWrapper(feat_dim)
            elif cfg.cost.model == "croco_dpt":
                self.nn = CroCoDPTWrapper(feat_dim, both_sided=cfg.cost.both_sided)

    def get_feats(self, batch, src_idx=[2], tgt_idx=[1]):
        if self.method == "feats":
            src_feats = torch.stack([self.feats[:, s] for s in src_idx], dim=1)
            tgt_feats = torch.stack([self.feats[:, t] for t in tgt_idx], dim=1)
        elif self.method == "nn":
            B = batch.visible_src.shape[1]
            N = len(src_idx)

            src_in = torch.stack([batch.visible_src[s] for s in src_idx], dim=1)
            tgt_in = torch.stack([batch.visible_tgt[t] for t in tgt_idx], dim=1)
            src_in = torch.reshape(src_in, (src_in.shape[0] * src_in.shape[1], *src_in.shape[2:]))
            tgt_in = torch.reshape(tgt_in, (tgt_in.shape[0] * tgt_in.shape[1], *tgt_in.shape[2:]))

            if self.border_handling == "pad_feats":
                t = self.weight_sl//2
                model_patch_size = self.nn.model.patch_embed.patch_size
                assert(model_patch_size[0] == model_patch_size[1])
                t_patch = math.ceil(t/model_patch_size[0]) * model_patch_size[0]

                src_in = torch.nn.functional.pad(src_in, (t_patch, t_patch, t_patch, t_patch), mode="constant")
                tgt_in = torch.nn.functional.pad(tgt_in, (t_patch, t_patch, t_patch, t_patch), mode="constant")

            src_in, tgt_in = self.nn(src_in, img2=tgt_in), self.nn(tgt_in, img2=src_in)
            
            """
            # ouroboros
            src_in_up = src_in_[..., t_patch:-t_patch, t_patch:-t_patch].detach()
            tru_tgt = tgt_in_[..., t_patch:-t_patch, t_patch:-t_patch].detach()
            src_in_up = torch.reshape(src_in_up, (in_frames.shape[0], len(src_idx), -1, *src_in_up.shape[2:]))
            tru_tgt = torch.reshape(tru_tgt, (in_frames.shape[0], len(src_idx), -1, *tru_tgt.shape[2:]))
            """

            if self.border_handling == "pad_feats":
                trim = t_patch - t
                if trim > 0:
                    src_in = src_in[..., trim:-trim, trim:-trim]
                if t_patch > 0:
                    tgt_in = tgt_in[..., t_patch:-t_patch, t_patch:-t_patch]

            src_feats = torch.reshape(src_in, (B, N, -1, *src_in.shape[2:]))
            tgt_feats = torch.reshape(tgt_in, (B, N, -1, *tgt_in.shape[2:]))
            src_feats = src_feats / (torch.linalg.norm(src_feats, dim=2, keepdim=True) + 1e-10)
            tgt_feats = tgt_feats / (torch.linalg.norm(tgt_feats, dim=2, keepdim=True) + 1e-10)

        return src_feats, tgt_feats


    def forward(self, batch, weights=None, temp=None, src_idx=[0], tgt_idx=[1]):
        B = batch.visible_src.shape[1]
        N = len(src_idx)
        device = batch.visible_src.device
        if temp is None:
            temp = 1.

        pred_occ_mask = None
        if weights is None:
            if self.method == 'weights':
                weights = self.weights
                neighbor_diff_x, neighbor_diff_y = 0., 0.
            elif self.method == 'feats' or self.method == 'nn':
                if self.method == 'feats' and self.border_handling == 'pad_feats':
                    raise ValueError("pad_feats border handling not supported for non-NN weight methods yet") 

                src_feats, tgt_feats = self.get_feats(batch, src_idx=src_idx, tgt_idx=tgt_idx)
                src_feats = torch.nn.functional.interpolate(src_feats[0], scale_factor=self.downsample_factor, mode='bilinear')[None]

                # Compute common fate loss (1)
                neighbor_diff_x = 1. - torch.sum(tgt_feats[..., :-1, :] * tgt_feats[..., 1:, :], dim=2)
                neighbor_diff_y = 1. - torch.sum(tgt_feats[..., :, :-1] * tgt_feats[..., :, 1:], dim=2)
                #

                src_feats = pad_for_filter(src_feats, self.weight_sl, self.downsample_factor, pad=self.border_handling != "pad_feats")
                tgt_feats = tgt_feats[..., None, None]

                weights = torch.sum(src_feats * tgt_feats, dim=2) # sum across the features

        weights_shape = weights.size()
        weights = weights.flatten(start_dim=-2) * temp
        weights = F.softmax(weights, dim=-1).view(weights_shape)

        if self.pred_occ_mask == "3fb":
            assert(self.downsample_factor == 1)
            weights_range = get_range(weights.detach())
            pred_occ_mask = torch.gt(weights_range, weights_range.new_tensor(0.5)).float()

        weights = weights.to(device)
        grid = self.grid.to(device)

        loss_src = torch.stack([batch.loss_src[s] for s in src_idx], dim=1)
        x_padded = pad_for_filter(loss_src, self.weight_sl, self.downsample_factor, border_fourth_channel=self.border_handling in ["fourth_channel", "pad_feats"])

        # For each pixel in the input frame, compute the weighted sum of its neighborhood
        out = torch.einsum('bnijkl, bncijkl->bncij', weights, x_padded)
        expected_positions = torch.einsum('bnijkl,klm->bnijm', weights, grid) / self.downsample_factor

        expected_positions = torch.permute(expected_positions, (0, 1, 4, 2, 3))
        expected_positions = expected_positions.flip(2)

        return {
                    'out':out, 
                    'flow': expected_positions, 
                    'weights': weights, 
                    'loss_src': loss_src,
                    'loss_tgt': torch.stack([batch.loss_tgt[t] for t in tgt_idx], dim=1),
                    'pred_occ_mask': pred_occ_mask,
                    'neighbor_diff': (neighbor_diff_x, neighbor_diff_y),
        }

class OverfitSoftLearner(L.LightningModule):
    def __init__(self, cfg: DictConfig, val_dataset=None):
        super().__init__()

        self.cfg = cfg
        self.downsample_factor = cfg.model.downsample_factor
        self.filter_zoom = cfg.model.filter_zoom
        self.inference_mode = cfg.model.inference_mode
        self.temp = torch.nn.Parameter(torch.Tensor([cfg.model.temp]))

        imsz_sm = [int(x) for x in cfg.dataset.imsz.split(",")]
        imsz_lg = [int(x) for x in cfg.dataset.imsz_super.split(",")]
        assert(imsz_sm[0] * self.downsample_factor == imsz_lg[0])
        assert(imsz_sm[1] * self.downsample_factor == imsz_lg[1])

        self.model = SoftMultiFramePredictor(cfg, 3, downsample_factor=self.downsample_factor, weight_sl=cfg.model.weight_sl, filter_zoom=self.filter_zoom, feat_dim=cfg.model.feat_dim)

        self.automatic_optimization = False

        self.last_full_val = self.global_step
        self.val_dataset = deepcopy(val_dataset)
        if cfg.training.overfit_batch == 1:
            self.val_dataset.frame_paths = [self.val_dataset.frame_paths[cfg.dataset.idx]]
            self.val_dataset.flow_paths = [self.val_dataset.flow_paths[cfg.dataset.idx]]
            self.val_dataset.mask_paths = [self.val_dataset.mask_paths[cfg.dataset.idx]]
            self.val_dataset.idx = 0

    def configure_optimizers(self):
        if self.cfg.cost.model == "croco_dpt":
            param_groups = get_parameter_groups(self.model.nn.head, weight_decay=0.05)
            model_opt = torch.optim.AdamW(param_groups, lr=self.cfg.training.lr, betas=(0.9, 0.95))
        else:
            model_opt = torch.optim.Adam(self.trainer.model.parameters(), lr=self.cfg.training.lr, weight_decay=self.cfg.training.weight_decay)
        temp_opt = torch.optim.Adam([self.temp], lr=self.cfg.training.temp_lr, weight_decay=self.cfg.training.weight_decay) # even if the LR is 0, it will move a little

        if self.cfg.training.lr > 0:
            high_lr = self.cfg.training.high_lr/self.cfg.training.lr
        else:
            high_lr = self.cfg.training.high_lr
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(model_opt, lr_lambda = lambda x: high_lr if x < self.cfg.training.high_lr_steps else 1)

        return (
            {
                "optimizer": model_opt,
                "lr_scheduler": {
                    "scheduler": warmup_scheduler,
                }
            },
            { "optimizer": temp_opt }
        )

    def chunk(self, x):
        bsz = x.shape[0]
        x[:, 0, 0, 0] = x[:, 0, 0, 0] * 0.95  # not completely white so wandb doesn't bug out
        return list(torch.chunk(x, bsz))

    def loss(self, outputs, occ_mask=None):
        if self.model.border_handling in ["fourth_channel", "pad_feats"]:
            border_weights = outputs["out"][:, :, -1, None].detach()
            outputs["out"] = outputs["out"][:, :, :-1]
        else:
            border_weights = torch.ones_like(outputs["out"])

        if self.inference_mode == "three_frames" and self.cfg.cost.pred_occ_mask == "3fb":
            loss = criterion(outputs["out"]*outputs["pred_occ_mask"], 
                             outputs["loss_tgt"]*outputs["pred_occ_mask"],
                             self.cfg.model.charbonnier_flatness)
        else:
            loss = criterion(outputs["out"], outputs["loss_tgt"], self.cfg.model.charbonnier_flatness)

        if self.inference_mode == "three_frames":
            if self.cfg.model.temp_on_occ:
                if self.cfg.cost.pred_occ_mask not in ["3fb", "gt"]:
                    raise ValueError

                occ_weights = 1. - outputs["pred_occ_mask"]
                if self.cfg.cost.pred_occ_mask == "3fb":
                    occ_weights = occ_weights * torch.gt(border_weights, border_weights.new_tensor(0.5)).float()

                #temp_smooth_reg = temporal_smoothness_loss(outputs['flow'][:, 0] * occ_weights[:, 0], r=None, flow_fields_b=outputs['positions'][:, 1].detach() * occ_weights[:, 0])
                #temp_smooth_reg += temporal_smoothness_loss(outputs['positions'][:, 1] * occ_weights[:, 1], r=None, flow_fields_b=outputs['positions'][:, 0].detach() * occ_weights[:, 1])
                #temp_smooth_reg *= 0.5
                temp_smooth_reg = temporal_smoothness_loss(outputs['flow'] * occ_weights, r=None, flow_fields_b=(outputs['flow'].detach() * occ_weights).flip(1))
                # ^^ May not work yet
            else:
                temp_smooth_reg = temporal_smoothness_loss(outputs['flow'], r=None)
            loss += self.cfg.model.temp_smoothness * temp_smooth_reg

        spat_smooth_reg = spatial_smoothness_loss(outputs['weights'], 
                                                  image=outputs['loss_tgt'], 
                                                  edge_weight=self.cfg.model.smoothness_edge, 
                                                  occ_mask=None)
        loss += spat_smooth_reg * self.cfg.model.smoothness * (self.model.true_sl ** 2)

        spat_smooth_reg = position_spat_smoothness(outputs['flow'], degree=2)
        loss += spat_smooth_reg * self.cfg.model.pos_smoothness

        # Common fate loss
        flow_diff_x = torch.linalg.norm(outputs['flow'][:, :, :, :-1, :] - outputs['flow'][:, :, :, 1:, :], dim=2).detach()
        flow_diff_y = torch.linalg.norm(outputs['flow'][:, :, :, :, :-1] - outputs['flow'][:, :, :, :, 1:], dim=2).detach()
        flow_diff_x, flow_diff_y = torch.exp(-flow_diff_x * 10), torch.exp(-flow_diff_y * 10)
        cf_loss = (flow_diff_x * outputs['neighbor_diff'][0]).mean() + (flow_diff_y * outputs['neighbor_diff'][1]).mean()
        loss += cf_loss * self.cfg.model.common_fate

        return loss

    def do_raft_prediction(self, frame1, frame2, smurf=False):
        if not smurf:
            model = raft_large(weights='C_T_V2')
        else:
            model = raft_smurf(checkpoint="smurf_sintel.pt")
        model.to(frame1.device)

        mem_limit_scale = 1 / max(frame1.shape[2] / 1024, frame1.shape[3] / 1024, 1) # scale down for memory limits
        def scale_dim_for_memory(x):
            return [8 * math.ceil(xx / 8 * mem_limit_scale) for xx in x]
        frame1_round = torchvision.transforms.functional.resize(frame1, scale_dim_for_memory(frame1.shape[2:]))
        frame2_round = torchvision.transforms.functional.resize(frame2, scale_dim_for_memory(frame2.shape[2:]))

        return model(frame1_round, frame2_round)[-1] / (mem_limit_scale * self.downsample_factor)

    def check_filter_size(self, frame2, frame3, gt_flow):
        if gt_flow is None:
            gt_flow = self.do_raft_prediction(frame2, frame3)
        else:
            gt_flow = torch.nan_to_num(gt_flow)
        flow_max = torch.max(torch.abs(gt_flow))

        if flow_max > self.model.max_size:
            raise ValueError(f'GT flow is too large--max flow is {flow_max} but max possible is {self.model.max_size}')
        else:
            return flow_max
    
    def on_save_checkpoint(self, checkpoint):
        # Keep only last checkpoint on wandb
        if self.logger is not None:
            if self.cfg.wandb.mode == "online":
                api = wandb.Api()
                entity, project, exp_id = self.cfg.wandb.entity, self.cfg.wandb.project, self.logger._id
                run = api.run(f"{entity}/{project}/{exp_id}")
                for artifact in run.logged_artifacts():
                    if len(artifact.aliases) == 0:
                        artifact.delete()

    def forward(self, batch, temp_lambda=1.):
        if self.inference_mode == "base":
            src_idx, tgt_idx = [2], [1]
        elif self.inference_mode == "three_frames":
            src_idx, tgt_idx = [2, 0], [1, 1]
        elif self.inference_mode == "bisided":
            src_idx, tgt_idx = [2, 1], [1, 2]
        temp = self.temp * temp_lambda

        if self.cfg.cost.pred_occ_mask == "3fb":
            assert(self.inference_mode == "three_frames")
            with torch.no_grad():
                outputs = self.model(batch,
                                     temp=temp,
                                     src_idx=tgt_idx,
                                     tgt_idx=src_idx)
                occ_mask = outputs["pred_occ_mask"].detach()
                del outputs

        outputs = self.model(batch,
                             temp=temp,
                             src_idx=src_idx,
                             tgt_idx=tgt_idx)

        if self.cfg.cost.pred_occ_mask == "3fb":
            outputs["pred_occ_mask"] = occ_mask

        return outputs

    def training_step(self, batch, batch_idx):
        if self.global_step > self.cfg.model.border_handling_on:
            self.model.border_handling = self.cfg.model.border_handling

        model_opt, temp_opt = self.optimizers()
        sched = self.lr_schedulers()
        model_opt.zero_grad()
        temp_opt.zero_grad()

        outputs = self(batch)
        loss = self.loss(outputs, occ_mask=None)

        B = batch.frames.shape[1]
        self.log('train/loss', loss, prog_bar=True)
        self.log_dict({
            "train/flow_fwd_min": torch.min(outputs["flow"]),
            "train/flow_fwd_max": torch.max(outputs["flow"]),
            "train/flow_fwd_mean": torch.mean(outputs["flow"]),
            "train/flow_fwd_std": torch.mean(torch.std(outputs["flow"], dim=(-1, -2))),
        }, batch_size=B)

        if batch.flow is not None:
            gt_flow = batch.flow[0]
            epe = compute_epe(batch.flow[0], 
                              outputs["flow"][:, 0], 
                              scale_to=batch.frames_up.shape[3:])

            self.log('train/epe', epe, prog_bar=True)
            self.log_dict({
                "train/flow_gt_min": torch.min(gt_flow),
                "train/flow_gt_max": torch.max(gt_flow),
                "train/flow_gt_mean": torch.mean(gt_flow),
                "train/flow_gt_std": torch.mean(torch.std(gt_flow, dim=(-1, -2)))
            }, batch_size=B)

        self.manual_backward(loss)
        model_opt.step()
        temp_opt.step()
        sched.step()

    def validation_step(self, batch, batch_idx):
        if self.global_step > self.cfg.model.border_handling_on:
            self.model.border_handling = self.cfg.model.border_handling
        B = batch.frames.shape[1]

        # Sanity checks
        check_flow = None if batch.flow is None else batch.flow[0]
        if self.inference_mode == "three_frames":
            flow_max = max(self.check_filter_size(batch.frames_up[1], batch.frames_up[2], None), self.check_filter_size(batch.frames_up[1], batch.frames_up[0], check_flow))
        else:
            flow_max = self.check_filter_size(batch.frames_up[-2], batch.frames_up[-1], check_flow)
        if self.temp < 0:
            print(f'Temperature ({self.temp}) is less than zero')

        with torch.no_grad():
            outputs = self(batch) 
            outputs_100 = self(batch, temp_lambda=100)
            print(f'Max flow: {flow_max}, max pred flow: {torch.max(torch.abs(outputs["flow"]))}')

            fwd_flow = outputs["flow"][:, 0]
            if self.inference_mode in ["three_frames", "bisided"]:
                bwd_flow = outputs["flow"][:, 1]
            else:
                bwd_flow = None
            gt_flow = batch.flow[0]

            loss = self.loss(outputs, occ_mask=None)

            raft_pred = self.do_raft_prediction(batch.frames_up[1], batch.frames_up[2])
            smurf_pred = self.do_raft_prediction(batch.frames_up[1], batch.frames_up[2], smurf=True)

            raft_epe = compute_epe(gt_flow, raft_pred)
            smurf_epe = compute_epe(gt_flow, smurf_pred)

            epe = compute_epe(gt_flow, 
                              fwd_flow, 
                              scale_to=batch.frames_up.shape[3:])
            mode_epe = compute_epe(gt_flow,
                              outputs_100['flow'][:, 0], 
                              scale_to=batch.frames_up.shape[3:])
            if "occ" in batch.get_dict("masks"):
                occ_epe = compute_epe(gt_flow * batch.get_dict("masks")["not_occ"], 
                                  fwd_flow * batch.get_dict("masks")["not_occ"], 
                                  scale_to=batch.frames_up.shape[3:])
            else:
                occ_epe = 0.

            self.log_dict({
                "val/loss": loss, 
                "val/raft_epe": raft_epe, 
                "val/smurf_epe": smurf_epe,
                "val/epe": epe,
                "val/mode_epe": mode_epe,
                "val/occ_epe": occ_epe,
                "temp": torch.mean(self.temp),
            }, sync_dist=True, batch_size=B)
            print(f"Current epe: {epe}")

            # Soft flow maps f1, f3 to f2
            if self.logger is None:
                return

            target = outputs["loss_tgt"][:, 0]
            if target.shape[1] > 3:
                target = target[:, :3]
            warped_imgs_fwd = outputs["out"][:, -1]
            if warped_imgs_fwd.shape[1] > 3:
                warped_imgs_fwd = warped_imgs_fwd[:, :3]

            # Flow visualizations
            batch_scale = torch.max(batch.frames[1])
            fwd_flows = torchvision.utils.flow_to_image(fwd_flow) / 255 * batch_scale
            fwd_flows_100 = torchvision.utils.flow_to_image(outputs_100["flow"][:, 0]) / 255 * batch_scale
            if bwd_flow is not None:
                bwd_flows = torchvision.utils.flow_to_image(bwd_flow) / 255 * batch_scale
            else:
                bwd_flows = torchvision.utils.flow_to_image(torch.zeros_like(fwd_flow)) / 255 * batch_scale
            gt_flows = torchvision.utils.flow_to_image(torch.nan_to_num(gt_flow)) / 255 * batch_scale
            diff_flows = torchvision.utils.flow_to_image(torch.nan_to_num(fwd_flow - gt_flow)) / 255 * batch_scale
            raft_pred = torchvision.utils.flow_to_image(raft_pred) / 255 * batch_scale
            smurf_pred = torchvision.utils.flow_to_image(smurf_pred) / 255 * batch_scale

            # Colorwheel
            U = torch.linspace(-1, 1, 100)
            V = torch.linspace(-1, 1, 100)
            X, Y = torch.meshgrid(U, V)
            wheel_flow = torch.stack((X, Y), dim=0)[None].to(batch.frames[1].device)
            wheel_flow = torchvision.utils.flow_to_image(wheel_flow) / 255 * torch.max(batch.frames[1])

            # Filters
            filters = filter_to_image(outputs["weights"][:, -1], downsample_factor=32)

            # Frames and flows
            fwd_flow = torch.cat((batch.frames[1], batch.frames[2], (batch.frames[1] + batch.frames[2])/2, fwd_flows), dim=3)
            fwd_warped = torch.cat((batch.frames[1], batch.frames[2], warped_imgs_fwd, 0.5 + 0.5 * (warped_imgs_fwd - batch.frames[1])), dim=3)
            gt_fwd = torch.cat((gt_flows, fwd_flows, diff_flows), dim=3)
            gt_fwd_100 = torch.cat((gt_flows, fwd_flows_100), dim=3)
            gt_fwd_raft = torch.cat((torchvision.transforms.functional.resize(gt_flows, (raft_pred.shape[2], raft_pred.shape[3])), raft_pred), dim=3)
            gt_fwd_smurf = torch.cat((torchvision.transforms.functional.resize(gt_flows, (smurf_pred.shape[2], smurf_pred.shape[3])), smurf_pred), dim=3)
            if outputs["pred_occ_mask"] is not None:
                occ_mask_vis = outputs["pred_occ_mask"][:, 0].repeat(1, 3, 1, 1)
            else:
                occ_mask_vis = torch.zeros_like(gt_flow[:, 0, None]).repeat(1, 3, 1, 1)

            # Log images
            self.logger.log_image(key='fwd_flow', images=self.chunk(fwd_flow), step=self.global_step) 
            self.logger.log_image(key='fwd_warped', images=self.chunk(fwd_warped), step=self.global_step) 
            self.logger.log_image(key='gt_fwd_flow', images=self.chunk(gt_fwd), step=self.global_step) 
            self.logger.log_image(key='gt_fwd_flow_100', images=self.chunk(gt_fwd_100), step=self.global_step) 
            self.logger.log_image(key='gt_raft', images=self.chunk(gt_fwd_raft), step=self.global_step) 
            self.logger.log_image(key='gt_smurf', images=self.chunk(gt_fwd_smurf), step=self.global_step) 
            self.logger.log_image(key='diff_flow', images=self.chunk(diff_flows), step=self.global_step) 
            self.logger.log_image(key='filters', images=self.chunk(filters), step=self.global_step) 
            self.logger.log_image(key='occ_mask', images=self.chunk(occ_mask_vis), step=self.global_step) 
            self.logger.log_image(key='colorwheel', images=self.chunk(wheel_flow), step=self.global_step) 

            # Full validation
            if self.cfg.validation.full_val_every is not None and self.global_step - self.last_full_val >= self.cfg.validation.full_val_every:
                self.last_full_val = self.global_step
                if self.val_dataset is not None:
                    import src.evaluation as evaluation
                    self.val_dataset.return_uncropped = True

                    # Get model predicted flow
                    def model_fwd(b):
                        b.to(batch.frames[0].device)
                        return tiled_pred(self.model, b, self.cfg.dataset.flow_max, lambda x, y: self.val_dataset.crop_batch(x, crop=y), crop=(224, 224), temp=self.temp)
                    full_out = evaluation.evaluate_against(self.val_dataset, model_fwd, evaluation.get_smurf_fwd(self.val_dataset, device=batch.frames[0].device), cfg=self.cfg)

                    self.logger.log_image(key="FULL_random", images=[full_out["random"]])
                    self.logger.log_image(key="FULL_worst", images=[full_out["worst"]])
                    self.logger.log_image(key="FULL_best", images=[full_out["best"]])

                    if "worst_comp" in full_out.keys():
                        self.logger.log_image(key="FULL_worst_comp", images=[full_out["worst_comp"]])
                    if "best_comp" in full_out.keys():
                        self.logger.log_image(key="FULL_best_comp", images=[full_out["best_comp"]])
                    self.log_dict({"FULL_" + k: v.nanmean().item() for k, v in full_out["metrics"].items()}, batch_size=batch.frames[0].shape[0])
                    print({"FULL_" + k: v.nanmean().item() for k, v in full_out["metrics"].items()})

                    # Get attention maps
                    def model_att(b):
                        b.to(batch.frames[0].device)
                        t_patch = math.ceil((self.model.weight_sl//2)/16)*16

                        def get_dprod(bb):
                            input_2 = torch.nn.functional.pad(bb.frames[2], (t_patch, t_patch, t_patch, t_patch), mode="constant")
                            input_1 = torch.nn.functional.pad(bb.frames[1], (t_patch, t_patch, t_patch, t_patch), mode="constant")
                            src_feats = self.model.nn(input_2, input_1)
                            tgt_feats = self.model.nn(input_1, input_2)
                            src_feats = src_feats / torch.linalg.norm(src_feats, dim=1, keepdim=True)
                            tgt_feats = tgt_feats / torch.linalg.norm(tgt_feats, dim=1, keepdim=True)

                            return torch.cat([src_feats[:, :, t_patch:-t_patch, t_patch:-t_patch], tgt_feats[:, :, t_patch:-t_patch, t_patch:-t_patch]], dim=1)

                        occ = tiled_pred(get_dprod, b, self.cfg.dataset.flow_max, lambda x, y: self.val_dataset.crop_batch(x, crop=y), crop=(224, 224), temp=self.temp, out_key='given', loss_fn=self.loss, given_dim=self.cfg.model.feat_dim * 2)
                        return occ
                    full_out_back = evaluation.evaluate_against(self.val_dataset, model_att, evaluation.get_smurf_fwd(self.val_dataset, device=batch.frames[0].device), cfg=self.cfg)
                    self.logger.log_image(key="FULL_dprod_random", images=[full_out_back["random"]])

                    # Print out feats
                    if self.cfg.cost.method in ['feats', 'nn']:
                        if self.cfg.cost.method == 'feats':
                            src_feats = self.model.feats[:, 2] / torch.linalg.norm(self.model.feats[:, 2], dim=2, keepdim=True)
                            tgt_feats = self.model.feats[:, 1] / torch.linalg.norm(self.model.feats[:, 1], dim=2, keepdim=True)
                        else:
                            src_feats, tgt_feats = self.model.get_feats(batch, src_idx=[2], tgt_idx=[1])
                        src_feats = (torch.permute(src_feats[:, 0], (1, 0, 2, 3)) * 0.5 + 0.5) * 255
                        tgt_feats = (torch.permute(tgt_feats[:, 0], (1, 0, 2, 3)) * 0.5 + 0.5) * 255
                        self.logger.log_image(key='src_feats', images=self.chunk(src_feats), step=self.global_step)
                        self.logger.log_image(key='tgt_feats', images=self.chunk(tgt_feats), step=self.global_step)
            elif self.cfg.validation.full_val_every is not None:
                print(f"{self.cfg.validation.full_val_every - (self.global_step - self.last_full_val)} steps until next fullval")

def compute_epe(gt, pred, scale_to=None):
    _, _, h_pred, w_pred = pred.size()
    bs, nc, h_gt, w_gt = gt.size()

    u_gt, v_gt = gt[:,0,:,:], gt[:,1,:,:]
    if scale_to is None:
        pred = torch.nn.functional.upsample(pred, size=(h_gt, w_gt), mode='bilinear')
    else:
        pred = torch.nn.functional.upsample(pred, size=scale_to, mode='bilinear')
        pred = torch.nn.functional.pad(pred, ((w_gt-scale_to[1])//2, (w_gt-scale_to[1]+1)//2, (h_gt-scale_to[0])//2, (h_gt-scale_to[0]+1)//2), mode='constant', value=float('nan'))
    u_pred = pred[:,0,:,:] * (w_gt/w_pred)
    v_pred = pred[:,1,:,:] * (h_gt/h_pred)

    epe = torch.sqrt(torch.pow((u_gt - u_pred), 2) + torch.pow((v_gt - v_pred), 2))

    resized_pred = torch.ones_like(pred)
    resized_pred[:,0,:,:] = u_pred
    resized_pred[:,1,:,:] = v_pred

    return epe.nanmean()
