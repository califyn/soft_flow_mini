import lightning as L
from omegaconf import DictConfig
import math
import functools

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
from data.superres import Batch

from models.croco import CroCoNet
from PIL import Image
import torchvision.transforms
from torchvision.transforms import ToTensor, Normalize, Compose

class EigenResult():
    def __init__(self, result):
        if result is not None:
            self.evecs = torch.Tensor(result[0])
            self.evals = result[1]
            self.mass = torch.Tensor(result[2])
            self.L = result[3]

            #self.to_spectral_mtx = self.evecs.T
            #self.to_original_mtx = torch.diag(self.mass) @ self.evecs
            self.to_spectral_mtx = self.evecs.T @ torch.diag(self.mass)
            self.to_original_mtx = self.evecs

            x = self.to_original_mtx @ self.to_spectral_mtx
            print(x)
            print(torch.mean(torch.diag(x)))

    def to_spectral(self, x):
        if isinstance(x, tuple):
            return tuple([self.to_spectral(xx) for xx in x])
        else:
            self.orig_shape = x.shape
            x = x.reshape((x.shape[0]*x.shape[1], x.shape[2]*x.shape[3]))
            #ret = torch.sum(self.to_spectral_mtx[None, :, :].to(x.device) * x[:, None], dim=2)
            ret = torch.matmul(self.to_spectral_mtx[None, :, :].to(x.device), x[:, :, None])[..., 0]

            ret = ret.reshape((self.orig_shape[0], self.orig_shape[1], -1))
            ret = ret.permute((0, 2, 1))

            return ret
            #return x
            #ret = torch.sum(self.to_spectral_mtx[None, :, :, None].to(x.device) * x[:, None], dim=2)
            #print(ret)
            #input([torch.sort(torch.abs(ret[..., i]), descending=True) for i in range(ret.shape[-1])])
            #return ret
            # (1, K, N, 1) * (B, 1, N, C)
            # (B, K, N, C)
            #x = x.reshape((x.shape[0], 176, 240, -1))
            #c = x.shape[-1]
            #x = torch.permute(x, (0, 3, 1, 2))
            """
            t = x - x.min()
            t = t / x.max()
            t = t[:, :3]
            torchvision.utils.save_image(t, "pre.png")
            """
            self.xy = (x.shape[2], x.shape[3])
            x = torch.fft.rfft2(x)
            x_pad = torch.zeros_like(x)
            #x_pad[:, :, :x.shape[2]//1, :x.shape[3]//1] += x[:, :, :x.shape[2]//1, :x.shape[3]//1]
            x_pad[:, :, :x.shape[2]//2, :x.shape[3]//2] += x[:, :, :x.shape[2]//2, :x.shape[3]//2]
            #x = torch.permute(x_pad, (0, 2, 3, 1))
            #x = x.reshape((x.shape[0], -1, c))
            return x_pad
    
    def to_original(self, x):
        if isinstance(x, tuple):
            return tuple([self.to_original(xx) for xx in x])
        else:
            x = x.reshape((self.orig_shape[0], -1, self.orig_shape[1]))
            x = x.permute((0, 2, 1))
            x = x.reshape((self.orig_shape[0]*self.orig_shape[1], -1))

            #ret = torch.sum(self.to_original_mtx[None, :, :].to(x.device) * x[:, None], dim=2)
            ret = torch.matmul(self.to_original_mtx[None, :, :].to(x.device), x[:, :, None])[..., 0]
            ret = ret.reshape(self.orig_shape)

            return ret
            #return x
            #return torch.sum(self.to_original_mtx[None, :, :, None].to(x.device) * x[:, None], dim=2)
            # (1, N, K, 1) * (B, 1, K, C)
            # (B, N, C)
            #x = x.reshape((-1, self.xy[0], self.xy[1], x.shape[-1]))
            #x = torch.permute(x, (0, 3, 1, 2))
            x = torch.fft.irfft2(x, s=self.xy)
            """
            t = x - x.min()
            t = t / x.max()
            t = t[:, :3]
            torchvision.utils.save_image(t, "post.png")
            """
            #x = torch.permute(x, (0, 2, 3, 1))
            #x = x.reshape((x.shape[0], -1, x.shape[-1]))
            return x

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

class LaplacianBlock(nn.Module):
    def __init__(self, eigs, dims, patchify=None, unpatchify=None):
        super(LaplacianBlock, self).__init__()
        self.eigs = eigs
        self.diffusion = eigenutils.LaplacianDiffusionBlock(dims)

        #self.model_patchify = patchify
        #self.model_unpatchify = unpatchify

    """
    def unpatchify(self, x):
        if isinstance(x, tuple):
            ret = [self.unpatchify(xx) for xx in x]
            return tuple([r[0] for r in ret]), tuple([r[1] for r in ret])
        else:
            x = self.model_unpatchify(x)
            x = torch.permute(x, (0, 2, 3, 1))
            h, w = x.shape[1], x.shape[2]
            x = torch.reshape(x, (x.shape[0], x.shape[1]*x.shape[2], x.shape[3]))
            return x, (h, w)

    def patchify(self, x, hw):
        if isinstance(x, tuple):
            return tuple([self.patchify(xx, hwhw) for xx, hwhw in zip(x, hw)])
        else:
            x = torch.reshape(x, (x.shape[0], hw[0], hw[1], x.shape[-1]))
            x = torch.permute(x, (0, 3, 1, 2))
            x = self.model_patchify(x)
            return x
    """

    def forward(self, x):
        return x, 0. 
        #x = self.block(*args)

        #x_, hw = self.unpatchify(x)
        k = self.eigs.to_spectral(x)
        x_ = self.eigs.to_original(k)
        #return x_

        diffused = self.diffusion(k, torch.Tensor(self.eigs.evals).to(k.device))
        diffused = self.eigs.to_original(diffused)

        # maintain high frequencies but penalize them
        return diffused + (x - x_), torch.nn.functional.mse_loss(x - x_, torch.zeros_like(x))

class IdentityBlock(nn.Module):
    def __init__(self, d):
        super(IdentityBlock, self).__init__()

    def forward(self, x, eigs):
        return x, 0.

class CroCoWrapper(torch.nn.Module):
    def __init__(self, feat_dim, eigs):
        super(CroCoWrapper, self).__init__()
        ckpt = torch.load('/home/califyn/croco/pretrained_models/CroCo_V2_ViTBase_SmallDecoder.pth', 'cpu')
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

        #out, _, _ = self.model(img2, img1)
        out, _, _ = self.model(img1, img1)
        return self.model.unpatchify(out, channels=self.feat_dim, h=img1.shape[-2]//16, w=img1.shape[-1]//16)


class SoftMultiFramePredictor(torch.nn.Module):
    def __init__(self, cfg, n_feats, weight_sl=31, downsample_factor=1, filter_zoom=1, init='all', feat_dim=256):
        super(SoftMultiFramePredictor, self).__init__()
        #image_size = [int(x)//16 for x in cfg.dataset.imsz.split(",")]
        image_size = [int(x) for x in cfg.dataset.imsz.split(",")]
        assert(weight_sl % 2 == 1)

        self.height = image_size[1]
        self.width = image_size[0]
        self.n_frames = n_feats
        self.weight_sl = weight_sl
        self.downsample_factor = downsample_factor
        self.filter_zoom = filter_zoom
        assert((weight_sl + downsample_factor - 1) % filter_zoom == 0)
        #assert((weight_sl + downsample_factor - 1) % (2 * downsample_factor) == downsample_factor)
        self.true_sl = (weight_sl + downsample_factor - 1) // filter_zoom

        if cfg.model.fft_filter is not None:
            self.fft_filter = cfg.model.fft_filter
            if self.fft_filter == "rect":
                fft = [int(x) for x in cfg.model.fft.split(",")]
                fft = [fft[1], fft[0]]
                self.fft = fft
            elif self.fft_filter == "butter":
                self.fft = cfg.model.fft
            elif self.fft_filter == "eigen":
                self.fft = 0
        else:
            self.fft = None
        #self.eigs = EigenResult(eigenutils.get_grid_eigs_cached(self.height, self.width, (self.height*self.width-1)//64)) # 2960
        self.eigs = EigenResult(None)

        self.method = cfg.model.cost_method
        if self.method == 'weights':
            self.weights = torch.nn.Parameter(torch.zeros(1, self.n_frames-1, self.height, self.width, self.true_sl, self.true_sl)) 
        elif self.method == 'feats':
            fill_val = (1/feat_dim) ** 0.5
            self.feats = torch.nn.Parameter(torch.full((1, self.n_frames, feat_dim, self.height, self.width), fill_val))
        elif self.method == 'nn':
            """
            self.nn = CroCoWrapper(feat_dim, self.eigs)

            patch_size = self.nn.model.patch_embed.patch_size[0]
            model_patchify = functools.partial(patchify, patch_size)
            model_unpatchify = functools.partial(unpatchify, patch_size, self.height, self.width)
            self.nn.model.enc_blocks = nn.ModuleList([
                LaplacianBlock(b, self.eigs, model_patchify, model_unpatchify) for b in self.nn.model.enc_blocks
                ])
            self.nn.model.dec_blocks = nn.ModuleList([
                LaplacianBlock(b, self.eigs, model_patchify, model_unpatchify) for b in self.nn.model.dec_blocks
                ])
            """
            self.nn = HalfUnet(
               16,
               IdentityBlock, #functools.partial(LaplacianBlock, self.eigs),
               channels = 1 * 3,
               out_dim = 1 * feat_dim,
               dim_mults = (1, 2, 4, 8),
            )

        t = self.weight_sl//2 + (self.downsample_factor - 1)/2
        x_positions = torch.linspace(-t, t, self.weight_sl + self.downsample_factor - 1)
        y_positions = torch.linspace(-t, t, self.weight_sl + self.downsample_factor - 1)
        grid_x, grid_y = torch.meshgrid(x_positions, y_positions)
        self.grid = torch.stack((grid_x, grid_y), dim=-1) 
        self.grid = torch.reshape(self.grid, (self.true_sl, self.filter_zoom, self.true_sl, self.filter_zoom, -1))
        self.grid = torch.mean(self.grid, dim=(1, 3))

        self.max_size = torch.max(torch.abs(self.grid)) / self.downsample_factor

    def _get_nd_butterworth_filter(
        self, shape, factor, dtype=torch.float64, squared_butterworth=True
    ):
        """Create a N-dimensional Butterworth mask for an FFT

        Parameters
        ----------
        shape : tuple of int
            Shape of the n-dimensional FFT and mask.
        factor : float
            Fraction of mask dimensions where the cutoff should be.
        order : float
            Controls the slope in the cutoff region.
        high_pass : bool
            Whether the filter is high pass (low frequencies attenuated) or
            low pass (high frequencies are attenuated).
        real : bool
            Whether the FFT is of a real (True) or complex (False) image
        squared_butterworth : bool, optional
            When True, the square of the Butterworth filter is used.

        Returns
        -------
        wfilt : ndarray
            The FFT mask.

        """
        order = 2.0 # default
        ranges = []
        for i, d in enumerate(shape):
            # start and stop ensures center of mask aligns with center of FFT
            axis = torch.arange(-(d - 1) // 2, (d - 1) // 2 + 1) / (d * factor)
            ranges.append(torch.fft.ifftshift(axis**2))
        # adjustment for real
        limit = d // 2 + 1
        ranges[-1] = ranges[-1][:limit]

        # q2 = squared Euclidean distance grid
        q2 = functools.reduce(torch.add, torch.meshgrid(*ranges, indexing="ij")) # sparse=True
        q2 = q2.to(dtype)
        q2 = torch.pow(q2, order)
        wfilt = 1 / (1 + q2)
        if not squared_butterworth:
            np.sqrt(wfilt, out=wfilt)
        return wfilt

    def fft_ifft(self, feats, mask_in):
        if self.fft_filter in ["rect", "butter"]:
            fft_feats = torch.fft.rfft2(feats)

            if self.fft_filter == "rect":
                x_range = torch.arange(fft_feats.shape[-2])[:, None]
                y_range = torch.arange(fft_feats.shape[-1])[None, :]
                mask = torch.logical_and(x_range < mask_in[0], y_range < mask_in[1])
            elif self.fft_filter == "butter": # butterworth filter
                mask = self._get_nd_butterworth_filter(feats.shape[-2:], mask_in, dtype=fft_feats.dtype)

            for _ in range(fft_feats.ndim - 2):
                mask = mask[None]
            fft_feats_masked = fft_feats * mask.to(fft_feats.device)
            ret = torch.fft.irfft2(fft_feats_masked, feats.shape[-2:])
        elif self.fft_filter == "eigen":
            orig_shape = feats.shape
            feats = feats.reshape((feats.shape[0]*feats.shape[1], *feats.shape[2:]))
            feats_spec = self.eigs.to_spectral(feats)
            ret = self.eigs.to_original(feats_spec)
            ret = ret.reshape(orig_shape)

        return ret

    def forward(self, batch, weights=None, temp=None, src_idx=0, tgt_idx=1):
        in_frames = torch.stack(batch.frames_up, dim=1)
        in_down = torch.stack(batch.frames, dim=1)
        assert(in_frames.shape[0] == 1)

        src_in_up = torch.stack([in_frames[:, s] for s in src_idx], dim=1)
        src_in = torch.stack([in_down[:, s] for s in src_idx], dim=1)
        tgt_in = torch.stack([in_down[:, t] for t in tgt_idx], dim=1)
        loss = 0.0

        if weights is not None:
            weights = weights.flatten(start_dim=-2)
        else:
            if self.method == 'weights':
                weights = self.weights[:, src_idx, None]
                weights *= temp
                weights_shape = self.weights.size()
                weights = self.weights.flatten(start_dim=-2)
            elif self.method == 'feats' or self.method == 'nn':
                if in_frames.shape[1] == 1:
                    if self.method == "feats":
                        src_feats = self.feats[:, src_idx, None]
                    elif self.method == "nn":
                        src_feats, l = self.nn(in_frames[:, src_idx])
                        loss += l
                else:
                    if self.method == "feats":
                        src_feats = torch.stack([self.feats[:, s] for s in src_idx], dim=1)
                    elif self.method == "nn":
                        src_in_ = torch.reshape(src_in, (src_in.shape[0] * src_in.shape[1], *src_in.shape[2:]))
                        tgt_in_ = torch.reshape(tgt_in, (tgt_in.shape[0] * tgt_in.shape[1], *tgt_in.shape[2:]))
                        src_in_, l = self.nn(src_in_)
                        tgt_in_, l = self.nn(tgt_in_)
                        #out, l = self.nn(torch.concatenate((src_in_, tgt_in_), dim=1))
                        #src_in_, tgt_in_ = torch.chunk(out, 2, dim=1)
                        src_feats = torch.reshape(src_in_, (in_frames.shape[0], len(src_idx), -1, *src_in.shape[3:]))
                        loss += l
                if self.fft is not None:
                    src_feats = self.fft_ifft(src_feats, self.fft)
                src_feats = torch.nn.functional.interpolate(src_feats[0], scale_factor=self.downsample_factor, mode='bilinear')[None]
                src_feats = src_feats / (torch.linalg.norm(src_feats, dim=2, keepdim=True) + 1e-10)
                src_feats = pad_for_filter(src_feats, self.weight_sl, self.downsample_factor)

                if in_frames.shape[1] == 1:
                    if self.method == "feats":
                        tgt_feats = self.feats[:, tgt_idx, None]
                    elif self.method == "nn":
                        tgt_feats, l = self.nn(in_frames[:, tgt_idx])
                        loss += l
                else:
                    if self.method == "feats":
                        tgt_feats = torch.stack([self.feats[:, t] for t in tgt_idx], dim=1)
                    elif self.method == "nn":
                        tgt_feats = torch.reshape(tgt_in_, (in_frames.shape[0], len(tgt_idx), -1, *tgt_in.shape[3:]))
                        loss += l
                if self.fft is not None:
                    tgt_feats = self.fft_ifft(tgt_feats, self.fft)
                tgt_feats = tgt_feats / (torch.linalg.norm(tgt_feats, dim=2, keepdim=True) + 1e-10)
                tgt_feats = tgt_feats[..., None, None]

                #print(torch.max(torch.abs(src_feats)), torch.min(torch.abs(src_feats)), torch.max(torch.abs(tgt_feats)), torch.min(torch.abs(tgt_feats)))
                weights = torch.sum(src_feats * tgt_feats, dim=2) # sum across the features
                weights *= temp
                weights_shape = weights.size()
                weights = weights.flatten(start_dim=-2)
        if temp is None:
            temp = 1.
        weights = F.softmax(weights, dim=-1).view(weights_shape)

        weights = weights.to(in_frames.device)
        grid = self.grid.to(in_frames.device)

        x_padded = pad_for_filter(src_in_up, self.weight_sl, self.downsample_factor)

        # For each pixel in the input frame, compute the weighted sum of its neighborhood
        out = torch.einsum('bnijkl, bncijkl->bncij', weights, x_padded)
        expected_positions = torch.einsum('bnijkl,klm->bnijm', weights, grid) / self.downsample_factor
        expected_norm = torch.einsum('bnijkl,kl->bnij', weights, grid.norm(dim=-1)) / self.downsample_factor  # Shape: [height, width, 2]
        expected_positions = expected_positions.flip(-1)

        return {
                    'out':out, 
                    'positions': expected_positions, 
                    'weights':weights, 
                    'exp_norm':expected_norm, 
                    'src': src_in, 
                    'src_up': src_in_up,
                    'tgt': tgt_in,
                    'loss': loss
                }

    def set(self, idx, set_to):
        self.weights.data[:, idx] = torch.log(torch.maximum(set_to, torch.full_like(set_to, 1e-24)))

class OverfitSoftLearner(L.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg
        self.downsample_factor = cfg.model.downsample_factor
        self.filter_zoom = cfg.model.filter_zoom
        self.set_model = True
        self.three_frames = cfg.model.three_frames
        self.weight_sl = cfg.model.weight_sl
        self.temp = torch.nn.Parameter(torch.Tensor([cfg.model.temp]))
        self.bisided = cfg.model.bisided

        imsz_sm = [int(x) for x in cfg.dataset.imsz.split(",")]
        imsz_lg = [int(x) for x in cfg.dataset.imsz_super.split(",")]
        assert(imsz_sm[0] * self.downsample_factor == imsz_lg[0])
        assert(imsz_sm[1] * self.downsample_factor == imsz_lg[1])

        if self.three_frames:
            self.model = SoftMultiFramePredictor(cfg, 3, downsample_factor=self.downsample_factor, weight_sl=self.weight_sl, filter_zoom=self.filter_zoom, feat_dim=cfg.model.feat_dim) # 123 for num. 11
        else:
            if self.bisided and self.cfg.model.cost_method == "weights":
                self.model = SoftMultiFramePredictor(cfg, 3, downsample_factor=self.downsample_factor, weight_sl=self.weight_sl, filter_zoom=self.filter_zoom, feat_dim=cfg.model.feat_dim) # 123 for num. 11
            else:
                #self.model = SoftMultiFramePredictor(cfg, 2, downsample_factor=self.downsample_factor, weight_sl=self.weight_sl, filter_zoom=self.filter_zoom, feat_dim=cfg.model.feat_dim) # 123 for num. 11
                self.model = SoftMultiFramePredictor(cfg, 3, downsample_factor=self.downsample_factor, weight_sl=self.weight_sl, filter_zoom=self.filter_zoom, feat_dim=cfg.model.feat_dim) # 123 for num. 11

        self.old_epe_full = None
        self.automatic_optimization = False

    def configure_optimizers(self):
        model_opt = torch.optim.Adam(self.trainer.model.parameters(), lr=self.cfg.training.lr, weight_decay=self.cfg.training.weight_decay)
        temp_opt = torch.optim.Adam([self.temp], lr=5e-1, weight_decay=self.cfg.training.weight_decay)

        return model_opt, temp_opt

    def chunk(self, x):
        bsz = x.shape[0]
        x[:, 0, 0, 0] = x[:, 0, 0, 0] * 0.95  # not completely white so wandb doesn't bug out
        return list(torch.chunk(x, bsz))

    def loss(self, outputs, occ_mask=None):
        """
        if occ_mask is not None:
            mult_factor = (~occ_mask.repeat(1, 1, 3, 1, 1)).float()
        else:
            mult_factor = torch.ones_like(targets["out"])
        """

        # Photometric
        #loss, _ = normal_minperblob(outputs['weights'], outputs["src_up"], outputs["tgt"], self.model.weight_sl, self.model.downsample_factor, self.model.filter_zoom, flatness=self.cfg.model.charbonnier_flatness)
        if occ_mask is not None:
            loss = criterion(outputs["out"]*(1-occ_mask), outputs["tgt"]*(1-occ_mask), self.cfg.model.charbonnier_flatness)/torch.mean(1-occ_mask)
        else:
            loss = criterion(outputs["out"], outputs["tgt"], self.cfg.model.charbonnier_flatness)

        if self.three_frames:
            temp_smooth_reg = temporal_smoothness_loss(outputs['positions'])
            #temp_smooth_reg = full_temporal_smoothness_loss(outputs['weights'])
            loss += self.cfg.model.temp_smoothness * temp_smooth_reg

        if occ_mask is not None:
            occ_mask = occ_mask.float()
        #spat_smooth_reg = spatial_smoothness_loss(outputs['weights'], image=outputs['tgt'], occ_mask=occ_mask)
        spat_smooth_reg = spatial_smoothness_loss(outputs['weights'], image=None, occ_mask=None)
        loss += spat_smooth_reg * self.cfg.model.smoothness * (self.model.true_sl ** 2) # finally weighing this #2.8e4 #* (61/31) * (61/31) # for sintel

        spat_smooth_reg = position_spat_smoothness(outputs['positions'])
        loss += spat_smooth_reg * self.cfg.model.pos_smoothness #2e-3

        norm_reg = outputs['exp_norm'].mean()
        loss += norm_reg * self.cfg.model.norm

        entropy_reg = entropy_loss(outputs['weights'])
        loss += entropy_reg * self.cfg.model.entropy

        #loss += 0.0 * outputs['loss']

        #loss += 5e-5 * bijectivity_loss(outputs, downsample_factor=self.downsample_factor)

        return loss

    def check_filter_size(self, frame2, frame3, gt_flow):
        if gt_flow is None: # no gt flow
            raft = raft_large(weights='C_T_V2')
            raft.to(frame2.device)

            mem_limit_scale = 1 / max(frame2.shape[2] / 1024, frame2.shape[3] / 1024, 1) # scale down for memory limits
            frame2_round = torchvision.transforms.functional.resize(frame2, (8 * math.ceil(frame2.shape[2] / 8 * mem_limit_scale), 8 * math.ceil(frame2.shape[3] / 8 * mem_limit_scale)))
            frame3_round = torchvision.transforms.functional.resize(frame3, (8 * math.ceil(frame3.shape[2] / 8 * mem_limit_scale), 8 * math.ceil(frame3.shape[3] / 8 * mem_limit_scale)))
            gt_flow = raft(frame2_round, frame3_round)[-1]

            flow_max = torch.max(torch.abs(gt_flow)) / (mem_limit_scale * self.downsample_factor)
        else:
            flow_max = torch.max(torch.abs(torch.nan_to_num(gt_flow)))

        if flow_max > self.model.max_size:
            raise ValueError(f'GT flow is too large--max flow is {flow_max} but max possible is {self.model.max_size}')
        else:
            return flow_max

    def training_step(self, batch, batch_idx):
        model_opt, temp_opt = self.optimizers()
        model_opt.zero_grad()
        temp_opt.zero_grad()

        if self.bisided:
            assert(not self.three_frames)
            outputs = self.model(batch,#Batch.index(batch, [2, 1]), 
                                 temp=self.temp,  
                                 src_idx=[2,1],
                                 tgt_idx=[1,2])
        elif self.three_frames:
            outputs = self.model(batch,
                                 temp=self.temp, 
                                 src_idx=[2,0],
                                 tgt_idx=[1,1])
        else:
            outputs = self.model(batch,
                                 src_idx=[2],
                                 tgt_idx=[1],
                                 temp=self.temp) 

        fwd_flows = [outputs["positions"][:, -1].permute(0, 3, 1, 2).float()]

        # this part only works with 2 frames
        #if self.three_frames:
        #    """
        #    swap_flows = torch.flip(outputs["positions"] - target["positions"], dims=[1]).permute(0, 1, 4, 2, 3).float()
        #    swap_flows = torch.reshape(swap_flows, (swap_flows.shape[0] * 2, *swap_flows.shape[2:]))
        #    occ_mask = softsplat(torch.zeros_like(-swap_flows), -swap_flows, None, strMode="sum")[:, -1, None] # self occ
        #    occ_mask = torch.reshape(occ_mask, (occ_mask.shape[0] // 2, 2, *occ_mask.shape[1:]))
        #    occ_mask = (occ_mask > 1.2).detach()
        #    """
        #    occ_mask = None
        #else:
        #    occ_mask = None
        #matched = transpose_filter(torch.flip(outputs["weights"], dims=[1]), downsample_factor=self.cfg.model.downsample_factor)
        #occ_mask = (torch.sum(matched, dim=(-2, -1)) < 0.8).float()[:, :, None].detach()
        #print(occ_mask.mean())
        #occ_mask = None

        loss = self.loss(outputs, occ_mask=None)
        fullres_fwd_flow = fwd_flows[-1] # correction deleted for now
        batch_flow = torch.Tensor([0]) if batch.flow is None else torch.stack(batch.flow, dim=1)
        self.log_dict({
            "train/loss": loss,
            "train/flow_fwd_min": torch.min(fullres_fwd_flow),
            "train/flow_fwd_max": torch.max(fullres_fwd_flow),
            "train/flow_fwd_mean": torch.mean(fullres_fwd_flow),
            "train/flow_fwd_std": torch.mean(torch.std(fullres_fwd_flow, dim=0)),
            "train/flow_gt_min": torch.min(batch_flow),
            "train/flow_gt_max": torch.max(batch_flow),
            "train/flow_gt_mean": torch.mean(batch_flow),
            "train/flow_gt_std": torch.mean(torch.std(batch_flow, dim=0))
        }, batch_size=batch.frames[0].shape[0])

        self.log('loss', loss, prog_bar=True)

        self.manual_backward(loss)
        model_opt.step()
        temp_opt.step()

    def validation_step(self, batch, batch_idx):
        check_flow = None if batch.flow is None else batch.flow[0]
        if self.three_frames:
            flow_max = max(self.check_filter_size(batch.frames_up[1], batch.frames_up[2], check_flow), self.check_filter_size(batch.frames_up[1], batch.frames_up[0], check_flow))
        else:
            flow_max = self.check_filter_size(batch.frames_up[-2], batch.frames_up[-1], check_flow)
        if self.temp < 0:
            print(f'Temperature ({self.temp}) is less than zero')

        raft = raft_large(weights='C_T_V2')
        raft.to(batch.frames[1].device)
        smurf = raft_smurf(checkpoint="smurf_sintel.pt")
        smurf.to(batch.frames[1].device)

        # Occ mask estimation
        mem_limit_scale = 1 / max(batch.frames_up[1].shape[2] / 1024, batch.frames_up[1].shape[3] / 1024, 1) # scale down for memory limits
        frame2_up_round = torchvision.transforms.functional.resize(batch.frames_up[1], (8 * math.ceil(batch.frames_up[1].shape[2] / 8 * mem_limit_scale), 8 * math.ceil(batch.frames_up[1].shape[3] / 8 * mem_limit_scale)))
        frame3_up_round = torchvision.transforms.functional.resize(batch.frames_up[2], (8 * math.ceil(batch.frames_up[2].shape[2] / 8 * mem_limit_scale), 8 * math.ceil(batch.frames_up[2].shape[3] / 8 * mem_limit_scale)))
        raft_pred_full_fwd = raft(frame2_up_round, frame3_up_round)[-1]
        raft_pred_full_bwd = raft(frame3_up_round, frame2_up_round)[-1]
        gt_flow_fwd = torch.nn.functional.interpolate(raft_pred_full_fwd, size=(batch.frames[1].shape[2], batch.frames[1].shape[3]))
        gt_flow_fwd = gt_flow_fwd / self.downsample_factor
        gt_flow_bwd = torch.nn.functional.interpolate(raft_pred_full_bwd, size=(batch.frames[1].shape[2], batch.frames[1].shape[3]))
        gt_flow_bwd = gt_flow_bwd / self.downsample_factor
        occ_mask_fwd = softsplat(torch.zeros_like(gt_flow_bwd), gt_flow_bwd, None, strMode="sum")[:, -1, None]
        occ_mask_bwd = softsplat(torch.zeros_like(gt_flow_fwd), gt_flow_fwd, None, strMode="sum")[:, -1, None]
        occ_mask_fwd = occ_mask_fwd < 0.8
        occ_mask_bwd = occ_mask_bwd < 0.8
        self.occ_mask = torch.stack((occ_mask_fwd, occ_mask_bwd), dim=1).float()[:, :, None].detach()
        # Occ mask estimation

        if not self.set_model:
            #self.model.set(1, flow_to_filter(gt_flow, self.model.weight_sl))
            #self.set_model = True
            raise NotImplementedError # not compatible rn

        with torch.no_grad():
            if self.bisided:
                assert(not self.three_frames)
                """
                outputs = self.model(Batch.index(batch, [2, 1]), 
                                     temp=self.temp,  
                                     src_idx=[0,1],#2,1
                                     tgt_idx=[1,0])#1,2
                outputs_100 = self.model(Batch.index(batch, [2, 1]), 
                                     temp=self.temp * 100,  
                                     src_idx=[0,1],#2,1
                                     tgt_idx=[1,0])#1,2
                """
                outputs = self.model(batch,
                                     temp=self.temp, 
                                     src_idx=[2,1],
                                     tgt_idx=[1,2])
                outputs_100 = self.model(batch,
                                     temp=self.temp * 100, 
                                     src_idx=[2,1],
                                     tgt_idx=[1,2])
            elif self.three_frames:
                outputs = self.model(batch,
                                     temp=self.temp, 
                                     src_idx=[2,0],
                                     tgt_idx=[1,1])
                outputs_100 = self.model(batch,
                                     temp=self.temp * 100, 
                                     src_idx=[2,0],
                                     tgt_idx=[1,1])
            else:
                outputs = self.model(batch,
                                     temp=self.temp,
                                     src_idx=[2],
                                     tgt_idx=[1]) 
                outputs_100 = self.model(batch,
                                     temp=self.temp * 100,
                                     src_idx=[2],
                                     tgt_idx=[1])

            fwd_flows = [outputs["positions"] [:, 0].permute(0, 3, 1, 2).float()]
            fwd_flows_100 = [outputs_100["positions"][:, 0].permute(0, 3, 1, 2).float()]
            if self.three_frames or self.bisided:
                bwd_flows = [outputs["positions"][:, 1].permute(0, 3, 1, 2).float()]
            else:
                bwd_flows = [torch.zeros_like(fwd_flows[-1]).float()]
            print(f'Max flow: {flow_max}, max pred flow: {torch.max(torch.abs(fwd_flows[-1]))}')

            loss = self.loss(outputs, occ_mask=None)

            with torch.set_grad_enabled(True):
                """
                #weights = self.model.weights.detach().clone()
                weights = self.model.weights
                weights.requires_grad_(True)
                outputs_grad = self.model(frame3_up[:, None], weights=weights)

                loss = self.loss(outputs_grad, target_outputs)
                loss.backward()

                grad_fwd_flow = -weights.grad[:, -1].clone() # negative; reflects the drcn of grad descent
                grad_fwd_flow = flow_from_weights(grad_fwd_flow)
                try:
                    grad_bwd_flow = -weights.grad[:, 0].clone() 
                    grad_bwd_flow = flow_from_weights(grad_bwd_flow)
                except:
                    grad_bwd_flow = torch.zeros_like(bwd_flows[-1])

                weights.grad = None
                #weights.requires_grad_(False)
                """
                grad_fwd_flow = torch.zeros_like(fwd_flows[-1])
                grad_bwd_flow = torch.zeros_like(bwd_flows[-1])

            mem_limit_scale = 1 / max(batch.frames_up[1].shape[2] / 1024, batch.frames_up[1].shape[3] / 1024, 1) # scale down for memory limits
            frame2_round = torchvision.transforms.functional.resize(batch.frames[1], (8 * math.ceil(batch.frames[1].shape[2] / 8), 8 * math.ceil(batch.frames[1].shape[3] / 8)))
            frame3_round = torchvision.transforms.functional.resize(batch.frames[2], (8 * math.ceil(batch.frames[2].shape[2] / 8), 8 * math.ceil(batch.frames[2].shape[3] / 8)))
            frame2_up_round = torchvision.transforms.functional.resize(batch.frames_up[1], (8 * math.ceil(batch.frames_up[1].shape[2] / 8 * mem_limit_scale), 8 * math.ceil(batch.frames_up[1].shape[3] / 8 * mem_limit_scale)))
            frame3_up_round = torchvision.transforms.functional.resize(batch.frames_up[2], (8 * math.ceil(batch.frames_up[2].shape[2] / 8 * mem_limit_scale), 8 * math.ceil(batch.frames_up[2].shape[3] / 8 * mem_limit_scale)))
            
            if min(frame2_round.shape[2], frame2_round.shape[3]) <= 128: # too small bugs out raft
                raft_pred = torch.zeros_like(frame2_round)[:, :2]
                smurf_pred = torch.zeros_like(frame2_round)[:, :2]
            else:
                raft_pred = raft(frame2_round, frame3_round)[-1]
                smurf_pred = smurf(frame2_round, frame3_round)[-1]
            if min(frame2_up_round.shape[2], frame2_up_round.shape[3]) <= 128: # too small bugs out raft
                raft_pred_full = torch.zeros_like(frame2_up_round)[:, :2]
                smurf_pred_full = torch.zeros_like(frame3_up_round)[:, :2]
            else:
                raft_pred_full = raft(frame2_up_round, frame3_up_round)[-1]
                smurf_pred_full = smurf(frame2_up_round, frame3_up_round)[-1]
                raft_pred_full_bwd = raft(frame3_up_round, frame2_up_round)[-1]

            if torch.any(torch.isnan(raft_pred)):
                raft_pred = torch.where(torch.isnan(raft_pred), 0.0, raft_pred)
            if torch.any(torch.isnan(raft_pred_full)):
                raft_pred_full = torch.where(torch.isnan(raft_pred_full), 0.0, raft_pred_full)
            if torch.any(torch.isnan(raft_pred_full_bwd)):
                raft_pred_full_bwd = torch.where(torch.isnan(raft_pred_full_bwd), 0.0, raft_pred_full_bwd)
            if torch.any(torch.isnan(smurf_pred)):
                smurf_pred = torch.where(torch.isnan(smurf_pred), 0.0, smurf_pred)
            if torch.any(torch.isnan(smurf_pred_full)):
                smurf_pred_full = torch.where(torch.isnan(smurf_pred_full), 0.0, smurf_pred_full)

            if batch.flow is None:
                gt_flow_orig = raft_pred_full
                gt_flow = torch.nn.functional.interpolate(raft_pred_full, size=(batch.frames[1].shape[2], batch.frames[1].shape[3]))
                gt_flow = gt_flow / self.downsample_factor
            else:
                gt_flow_orig = batch.flow_orig[0]
                gt_flow = batch.flow[0]

            # EPE at resolution
            raft_epe = compute_epe(gt_flow_orig, raft_pred)
            smurf_epe = compute_epe(gt_flow_orig, smurf_pred)

            epe_full = compute_epe(gt_flow_orig, fwd_flows[-1], scale_to=batch.frames_up[0].shape[2:])
            mode_epe_full = compute_epe(gt_flow_orig, fwd_flows_100[-1], scale_to=batch.frames_up[0].shape[2:])
            raw_epe_full = compute_epe(gt_flow_orig, fwd_flows[-1]) # original epe
            raft_epe_full = compute_epe(gt_flow_orig, raft_pred_full)
            smurf_epe_full = compute_epe(gt_flow_orig, smurf_pred_full)

            """
            if self.three_frames:
                swap_flows = torch.flip(outputs["positions"] - target_outputs["positions"], dims=[1]).permute(0, 1, 4, 2, 3).float()
                swap_flows = torch.reshape(swap_flows, (swap_flows.shape[0] * 2, *swap_flows.shape[2:]))
                occ_mask = softsplat(torch.zeros_like(-swap_flows), -swap_flows, None, strMode="sum")[:, -1, None] # self occ
                occ_mask = torch.reshape(occ_mask, (occ_mask.shape[0] // 2, 2, *occ_mask.shape[1:]))
                occ_mask = occ_mask[:, -1]
                occ_mask = (occ_mask > 1.2).detach()
                occ_mask = None
                occ_epe = float('nan')
            else:
            occ_mask = softsplat(torch.zeros_like(raft_pred_full_bwd), raft_pred_full_bwd, None, strMode="sum")[:, -1, None]
            #occ_mask = 1 - torch.minimum(occ_mask, torch.Tensor(1).to(occ_mask.device)) # this is from the paper
            occ_mask = occ_mask < 0.8
            print(occ_mask.shape, gt_flow_orig.shape)
            input("?")
            masked_gt_flow_orig = torch.where(occ_mask.repeat(1, 2, 1, 1), torch.full_like(gt_flow_orig, float('nan')), gt_flow_orig)
            occ_epe = compute_epe(masked_gt_flow_orig, fwd_flows[-1])
            if False: #self.bisided:
                matched = transpose_filter(outputs["weights"][:, -1, None], downsample_factor=self.cfg.model.downsample_factor)
                #match_mask = 1 - torch.sum(matched, dim=(-2, -1))
                match_mask = (torch.sum(matched, dim=(-2, -1)) < 0.8).float()
                print(torch.mean(match_mask))
                match_mask = torch.clamp(match_mask, 0, 1)
                match_mask = match_mask[:, 0]
                match_mask = torch.nn.functional.upsample(match_mask[None], size=(occ_mask.shape[-2], occ_mask.shape[-1]), mode='bilinear')[0]
                conv_match_occ = match_mask * occ_mask
            else:
                match_mask = torch.zeros_like(occ_mask)
                conv_match_occ = torch.zeros_like(occ_mask)
            """
            occ_mask = torch.zeros_like(gt_flow_orig[:, 0, None])
            match_mask = torch.zeros_like(gt_flow_orig[:, 0, None])
            conv_match_occ = torch.zeros_like(gt_flow_orig[:, 0, None])
            occ_epe = 0.0

            self.log_dict({
                "val/loss": loss, # training loss
                "val/raft_epe": raft_epe, # EPE between (1) RAFT with downscaled f2, f3 as input (2) full resolution flow
                "val/smurf_epe": smurf_epe, # EPE between (1) SMURF with downscaled f2, f3 as input (2) full resolution flow
                "val/epe_full": epe_full, # EPE between (1) flow from the soft flow overfitting on downscaled inputs (2) full resolution flow
                "val/raw_epe_full": raw_epe_full, # previous EPE but w/o target filter adjustment (same if no target filters)
                "val/raft_epe_full": raft_epe_full, # EPE between (1) RAFT with full resolution f2, f3 as input (2) full resolution flow
                "val/smurf_epe_full": smurf_epe_full, # EPE between (1) SMURF with full resolution f2, f3 as input (2) full resolution flow
                "val/mode_epe_full": mode_epe_full, # EPE between (1) SMURF with full resolution f2, f3 as input (2) full resolution flow
                "val/occ_epe": occ_epe,
                "temp": torch.mean(self.temp),
            }, sync_dist=True, batch_size=batch.frames[0].shape[0])
            print(epe_full)
            #"val/var_mean": torch.mean(outputs["var"])
            target = outputs["tgt"][:, 0]

            fwd_flow = fwd_flows[-1]
            fwd_flow_100 = fwd_flows_100[-1]
            bwd_flow = bwd_flows[-1]

            warped_imgs_flow = warp(batch.frames[1], fwd_flow)
            warped_imgs_fwd = outputs["out"][:, -1]

            fwd_flows = torchvision.utils.flow_to_image(fwd_flow) / 255 * torch.max(batch.frames[1])
            fwd_flows_100 = torchvision.utils.flow_to_image(fwd_flow_100) / 255 * torch.max(batch.frames[1])
            bwd_flows = torchvision.utils.flow_to_image(bwd_flow) / 255 * torch.max(batch.frames[1])
            gt_flows = torchvision.utils.flow_to_image(torch.nan_to_num(gt_flow)) / 255 * torch.max(batch.frames[1])
            diff_flows = torchvision.utils.flow_to_image(torch.nan_to_num(fwd_flow - gt_flow)) / 255 * torch.max(batch.frames[1])
            gradf_flows = torchvision.utils.flow_to_image(grad_fwd_flow.float()) / 255 * torch.max(batch.frames[1])
            gradb_flows = torchvision.utils.flow_to_image(grad_bwd_flow.float()) / 255 * torch.max(batch.frames[1])
            raft_pred = torchvision.utils.flow_to_image(raft_pred) / 255 * torch.max(batch.frames[1])
            raft_pred_full = torchvision.utils.flow_to_image(raft_pred_full) / 255 * torch.max(batch.frames[1])
            smurf_pred = torchvision.utils.flow_to_image(smurf_pred) / 255 * torch.max(batch.frames[1])
            smurf_pred_full = torchvision.utils.flow_to_image(smurf_pred_full) / 255 * torch.max(batch.frames[1])

            U = torch.linspace(-1, 1, 100)
            V = torch.linspace(-1, 1, 100)
            X, Y = torch.meshgrid(U, V)
            wheel_flow = torch.stack((X, Y), dim=0)[None].to(batch.frames[1].device)
            wheel_flow = torchvision.utils.flow_to_image(wheel_flow) / 255 * torch.max(batch.frames[1])

            filters = filter_to_image(outputs["weights"][:, -1], downsample_factor=32)

            combined_frames = torch.cat((batch.frames[0], batch.frames[1], batch.frames[2]), dim=3)
            target = torch.cat((target, batch.frames[1], target-batch.frames[1]), dim=3)
            fwd_flow = torch.cat((batch.frames[1], batch.frames[2], fwd_flows), dim=3)
            bwd_flow = torch.cat((batch.frames[0], batch.frames[1], bwd_flows), dim=3)
            fwd_warped = torch.cat((batch.frames[1], batch.frames[2], warped_imgs_fwd, 0.5 + 0.5 * (warped_imgs_fwd - batch.frames[1])), dim=3)
            fwd_warped_flow = torch.cat((batch.frames[1], batch.frames[2], warped_imgs_flow, 0.5 + 0.5 * (warped_imgs_flow - batch.frames[1])), dim=3)
            gt_fwd = torch.cat((gt_flows, fwd_flows), dim=3)
            gt_fwd_100 = torch.cat((gt_flows, fwd_flows_100), dim=3)
            gt_fwd_raft = torch.cat((torchvision.transforms.functional.resize(gt_flows, (raft_pred.shape[2], raft_pred.shape[3])), raft_pred), dim=3)
            gt_fwd_raft_full = torch.cat((torchvision.transforms.functional.resize(gt_flows, (raft_pred_full.shape[2], raft_pred_full.shape[3])), raft_pred_full), dim=3)
            gt_fwd_smurf = torch.cat((torchvision.transforms.functional.resize(gt_flows, (smurf_pred.shape[2], smurf_pred.shape[3])), smurf_pred), dim=3)
            gt_fwd_smurf_full = torch.cat((torchvision.transforms.functional.resize(gt_flows, (smurf_pred_full.shape[2], smurf_pred_full.shape[3])), smurf_pred_full), dim=3)
            occ_mask_vis = torch.cat((torchvision.transforms.functional.resize(gt_flows, (occ_mask.shape[2], occ_mask.shape[3])), occ_mask.repeat(1, 3, 1, 1).float(), match_mask.repeat(1, 3, 1, 1).float(), conv_match_occ.repeat(1, 3, 1, 1).float()), dim=3)

            # Soft flow maps f1, f3 to f2
            if self.logger is not None:
                self.logger.log_image(key='combined_frames', images=self.chunk(combined_frames), step=self.global_step) # f1, f2, f3 combined
                self.logger.log_image(key='fwd_flow', images=self.chunk(fwd_flow), step=self.global_step) # f2, f3, then predicted forward flow from soft flows
                self.logger.log_image(key='bwd_flow', images=self.chunk(bwd_flow), step=self.global_step) # not used
                self.logger.log_image(key='fwd_warped', images=self.chunk(fwd_warped), step=self.global_step) # f2, f3, then the output of the soft flow (i.e. f3 mapped to f2)
                self.logger.log_image(key='fwd_warped_flow', images=self.chunk(fwd_warped_flow), step=self.global_step) # f2, f3, then f3 backward warped with the predicted flow onto f2
                self.logger.log_image(key='target', images=self.chunk(target), step=self.global_step) # f2 after target filter, f2, and the difference between the two
                self.logger.log_image(key='gt_fwd_flow', images=self.chunk(gt_fwd), step=self.global_step) # gt forward flow (downscaled), predicted flow
                self.logger.log_image(key='gt_fwd_flow_100', images=self.chunk(gt_fwd_100), step=self.global_step) # gt forward flow (downscaled), predicted flow
                self.logger.log_image(key='gt_raft', images=self.chunk(gt_fwd_raft), step=self.global_step) # gt forward flow (downscaled), RAFT output with downscaled inputs
                self.logger.log_image(key='gt_raft_full', images=self.chunk(gt_fwd_raft_full), step=self.global_step) # gt forward flow (full resolution), RAFT output with full resolution inputs
                self.logger.log_image(key='gt_smurf', images=self.chunk(gt_fwd_smurf), step=self.global_step) # gt forward flow (downscaled), RAFT output with downscaled inputs
                self.logger.log_image(key='gt_smurf_full', images=self.chunk(gt_fwd_smurf_full), step=self.global_step) # gt forward flow (full resolution), RAFT output with full resolution inputs
                self.logger.log_image(key='diff_flow', images=self.chunk(diff_flows), step=self.global_step) # take the difference between predicted flow and gt, and visualize (this is another flow)
                self.logger.log_image(key='grad_fwd_flow', images=self.chunk(gradf_flows), step=self.global_step) # gradient of the forward flow
                self.logger.log_image(key='grad_bwd_flow', images=self.chunk(gradb_flows), step=self.global_step) # not used
                self.logger.log_image(key='filters', images=self.chunk(filters), step=self.global_step) # visualizing the filters inside the soft flow (1 filter shown per 32x32 pixels)
                self.logger.log_image(key='exp_norm', images=self.chunk(outputs['exp_norm'][:, -1, None].repeat(1,3,1,1)), step=self.global_step) # grayscale image of the expected norm of the flow
                self.logger.log_image(key='occ_mask', images=self.chunk(occ_mask_vis), step=self.global_step) # grayscale image of the expected norm of the flow
                self.logger.log_image(key='colorwheel', images=self.chunk(wheel_flow), step=self.global_step) # grayscale image of the expected norm of the flow

                if self.cfg.model.cost_method in ['feats', 'nn']:
                    if self.cfg.model.cost_method == 'feats':
                        src_feats = self.model.feats[:, 0, None] / torch.linalg.norm(self.model.feats[:, 0, None], dim=2, keepdim=True)
                        tgt_feats = self.model.feats[:, 1, None] / torch.linalg.norm(self.model.feats[:, 1, None], dim=2, keepdim=True)
                    else:
                        src_feats = self.model.nn(batch.frames[2], batch.frames[1])[0][:, None]
                        tgt_feats = self.model.nn(batch.frames[1], batch.frames[2])[0][:, None]
                        #src_feats = self.model.nn(torch.cat((batch.frames[2], batch.frames[1]), dim=1))[0][:, None]
                        #src_feats, tgt_feats = torch.chunk(src_feats, 2, dim=2)
                        src_feats = src_feats / torch.linalg.norm(src_feats, dim=2, keepdim=True)
                        tgt_feats = tgt_feats / torch.linalg.norm(tgt_feats, dim=2, keepdim=True)
                    #src_atan = torchvision.utils.flow_to_image(src_feats[0, :])
                    #tgt_atan = torchvision.utils.flow_to_image(tgt_feats[0, :])
                    src_feats = (torch.permute(src_feats[0], (1, 0, 2, 3)) * 0.5 + 0.5) * 255
                    tgt_feats = (torch.permute(tgt_feats[0], (1, 0, 2, 3)) * 0.5 + 0.5) * 255
                    self.logger.log_image(key='src_feats', images=self.chunk(src_feats), step=self.global_step)
                    self.logger.log_image(key='tgt_feats', images=self.chunk(tgt_feats), step=self.global_step)
                    """
                    #self.logger.log_image(key='src_feats_atan', images=self.chunk(src_atan), step=self.global_step)
                    #self.logger.log_image(key='tgt_feats_atan', images=self.chunk(tgt_atan), step=self.global_step)
                    """
                    
                    """
                    if self.cfg.model.temp_scheduling:
                        # Temperature increases when EPE no longer increases.
                        if self.old_epe_full != None and self.old_epe_full - 0.01 < epe_full:
                            print(f"Old temp: {self.temp}")
                            self.temp *= 1.1
                            print(f"New temp: {self.temp}")
                    """
                    self.old_epe_full = epe_full

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

def warp(x, flo):
    """
    backward warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)
    output = nn.functional.grid_sample(x, vgrid, padding_mode='border')
    mask = torch.autograd.Variable(torch.ones(x.size()), requires_grad=False).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)

    mask[mask.data<0.9999] = 0
    mask[mask.data>0] = 1

    return output
