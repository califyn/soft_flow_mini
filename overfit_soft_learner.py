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

from soft_losses import *
from soft_utils import flow_from_weights, filter_to_image, flow_to_filter, warp_previous_flow, pad_for_filter
from softsplat_downsample import softsplat

import matplotlib.pyplot as plt

class SoftMultiFramePredictor(torch.nn.Module):
    def __init__(self, cfg, n_feats, weight_sl=31, downsample_factor=1, filter_zoom=1, init='all', feat_dim=256):
        super(SoftMultiFramePredictor, self).__init__()
        image_size = [int(x) for x in cfg.dataset.imsz.split(",")]
        image_size = [image_size[1], image_size[0]]
        assert(weight_sl % 2 == 1)

        self.height = image_size[0]
        self.width = image_size[1]
        self.n_frames = n_feats
        self.weight_sl = weight_sl
        self.downsample_factor = downsample_factor
        self.filter_zoom = filter_zoom
        assert((weight_sl + downsample_factor - 1) % filter_zoom == 0)
        self.true_sl = (weight_sl + downsample_factor - 1) // filter_zoom
        """
        if init == 'zero':
            assert(downsample_factor == 1)
            self.weights.data[:, :, :, :, (self.weight_sl-1)//2, (self.weight_sl-1)//2] += 10.0
        """
        if cfg.model.fft_filter is not None:
            self.fft_filter = cfg.model.fft_filter
            if self.fft_filter == "rect":
                fft = [int(x) for x in cfg.model.fft.split(",")]
                fft = [fft[1], fft[0]]
                self.fft = fft
            elif self.fft_filter == "butter":
                self.fft = cfg.model.fft
        else:
            self.fft = None

        self.method = cfg.model.cost_method
        if self.method == 'weights':
            self.weights = torch.nn.Parameter(torch.zeros(1, self.n_feats-1, self.height, self.width, self.true_sl, self.true_sl)) 
        elif self.method == 'feats':
            fill_val = (1/feat_dim) ** 0.5
            #self.src_feats = torch.nn.Parameter(torch.full((1, self.n_frames, feat_dim, self.height * self.downsample_factor, self.width * self.downsample_factor), fill_val))
            #self.src_feats = torch.nn.Parameter(torch.full((1, self.n_frames, feat_dim, self.height, self.width), fill_val))
            #self.src_feats = torch.nn.Parameter(torch.full((1, 1, feat_dim, self.height, self.width), fill_val))
            #self.tgt_feats = torch.nn.Parameter(torch.full((1, self.n_frames, feat_dim, self.height, self.width), fill_val))
            #print('restricting tgt feats to only be 1 frame')
            #self.tgt_feats = torch.nn.Parameter(torch.full((1, 1, feat_dim, self.height, self.width), fill_val))
            self.feats = torch.nn.Parameter(torch.full((1, n_feats, feat_dim, self.height, self.width), fill_val))
        elif self.method == 'nn':
            self.nn = None

        t = self.weight_sl//2 + (self.downsample_factor - 1)/2
        x_positions = torch.linspace(-t, t, self.weight_sl + self.downsample_factor - 1)
        y_positions = torch.linspace(-t, t, self.weight_sl + self.downsample_factor - 1)
        grid_x, grid_y = torch.meshgrid(x_positions, y_positions)
        self.grid = torch.stack((grid_x, grid_y), dim=-1) 
        self.grid = torch.reshape(self.grid, (self.true_sl, self.filter_zoom, self.true_sl, self.filter_zoom, -1))
        self.grid = torch.mean(self.grid, dim=(1, 3))

        self.max_size = torch.max(torch.abs(self.grid)) / self.downsample_factor

    def cat_pos_emb(self, inp):
        x_pos = torch.linspace(-1, 1, inp.shape[-2]).to(inp.device)
        y_pos = torch.linspace(-1, 1, inp.shape[-1]).to(inp.device)

        grid_x, grid_y = torch.meshgrid(x_pos, y_pos)
        return torch.cat((inp, grid_x[None, None], grid_y[None, None]), dim=1)

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

        return ret

    def forward(self, in_frames, tgt_frames, weights=None, temp=None, in_down=None, src_idx=0, tgt_idx=1):
        if weights is not None:
            weights = weights.flatten(start_dim=-2)
        else:
            if self.method == 'weights':
                assert(in_frames.shape[1] == 1 and tgt_frames.shape[1] == 1)
                weights *= temp
                weights_shape = self.weights.size()
                weights = self.weights.flatten(start_dim=-2)
            elif self.method == 'feats':
                if in_frames.shape[1] == 1:
                    src_feats = self.feats[0, src_idx, None]
                else:
                    src_feats = torch.stack([self.feats[0, s] for s in src_idx], dim=0)
                if self.fft is not None:
                    src_feats = self.fft_ifft(src_feats, self.fft)
                src_feats = torch.nn.functional.interpolate(src_feats, scale_factor=self.downsample_factor, mode='bilinear')[None]
                src_feats = src_feats / torch.linalg.norm(src_feats, dim=2, keepdim=True)
                src_feats = pad_for_filter(src_feats, self.weight_sl, self.downsample_factor)

                if in_frames.shape[1] == 1:
                    tgt_feats = self.feats[:, tgt_idx, None]
                else:
                    tgt_feats = torch.stack([self.feats[:, t] for t in tgt_idx], dim=1)
                if self.fft is not None:
                    tgt_feats = self.fft_ifft(tgt_feats, self.fft)
                tgt_feats = tgt_feats / torch.linalg.norm(tgt_feats, dim=2, keepdim=True)
                tgt_feats = tgt_feats[..., None, None]

                weights = torch.sum(src_feats * tgt_feats, dim=2) # sum across the features
                weights *= temp
                weights_shape = weights.size()
                weights = weights.flatten(start_dim=-2)
            elif self.method == 'nn':
                assert(in_frames.shape[1] == 1 and tgt_frames.shape[1] == 1)
                # broken for n_frames > 1 (and feats too i think)
                raise ValueError('nn not compatible with src/txt idx yet')
                """
                if flip:
                    src_feats = self.tgt_nn(tgt_frames[:, 0])
                else:
                    src_feats = self.src_nn(in_down[:, 0])
                """
                #src_feats = self.src_nn(
                src_feats = torch.nn.functional.interpolate(src_feats, scale_factor=self.downsample_factor, mode='bilinear')[None]
                src_feats = src_feats / torch.linalg.norm(src_feats, dim=2, keepdim=True)
                src_feats = pad_for_filter(src_feats, self.weight_sl, self.downsample_factor)

                if flip:
                    tgt_feats = self.nn(in_down[:, 0])[:, None]
                else:
                    tgt_feats = self.nn(tgt_frames[:, 0])[:, None]
                tgt_feats = tgt_feats / torch.linalg.norm(tgt_feats, dim=2, keepdim=True)
                tgt_feats = tgt_feats[..., None, None]

                weights = torch.sum(src_feats * tgt_feats, dim=2) # sum across the features
                weights *= temp
                weights_shape = weights.size()
                weights = weights.flatten(start_dim=-2)
        if temp is None:
            temp = 1.
        weights = F.softmax(weights, dim=-1).view(weights_shape)

        weights = weights.to(in_frames.device)
        grid = self.grid.to(in_frames.device)

        """
        t = self.weight_sl//2 # smaller t for padding
        in_shape = in_frames.shape
        in_frames_reshaped = torch.reshape(in_frames, (in_frames.shape[0] * in_frames.shape[1], in_frames.shape[2], in_frames.shape[3], in_frames.shape[4]))
        x_padded = F.pad(in_frames_reshaped, (t, t, t, t), mode='replicate', value=0)
        x_padded = torch.reshape(x_padded, (in_shape[0], in_shape[1], *x_padded.shape[1:]))
        x_padded = x_padded.unfold(3, self.weight_sl + self.downsample_factor - 1, self.downsample_factor).unfold(4, self.weight_sl + self.downsample_factor - 1, self.downsample_factor)  # extract 30x30 patches around each pixel
        # should be perfect downsampling now
        """
        x_padded = pad_for_filter(in_frames, self.weight_sl, self.downsample_factor)

        # For each pixel in the input frame, compute the weighted sum of its neighborhood
        out = torch.einsum('bnijkl, bncijkl->bncij', weights, x_padded)
        #out = torch.zeros_like(x_padded[..., 0, 0]) # placeholder for now
        expected_positions = torch.einsum('bnijkl,klm->bnijm', weights, grid) / self.downsample_factor
        expected_norm = torch.einsum('bnijkl,kl->bnij', weights, grid.norm(dim=-1)) / self.downsample_factor  # Shape: [height, width, 2]
        expected_positions = expected_positions.flip(-1)

        #deviation = torch.sum(torch.square(expected_positions[:, :, :, :, None, None, :] - grid[None, None, None, None, :, :, :] / self.downsample_factor), dim=-1)
        #deviation = torch.sum(deviation * weights, dim=(4, 5))

        #return {'out':out, 'positions': expected_positions, 'input':in_frames, 'weights':weights, 'exp_norm':expected_norm, 'var': deviation}
        return {'out':out, 'positions': expected_positions, 'input':in_frames, 'weights':weights, 'exp_norm':expected_norm, 'in_down': in_down}

    def set(self, idx, set_to):
        self.weights.data[:, idx] = torch.log(torch.maximum(set_to, torch.full_like(set_to, 1e-24)))

class IdentityPredictor(torch.nn.Module):
    def __init__(self, cfg, n_frames, weight_sl=31):
        super(IdentityPredictor, self).__init__()
        image_size = [int(x) for x in cfg.dataset.imsz.split(",")]
        image_size = [image_size[1], image_size[0]]

        self.height = image_size[0]
        self.width = image_size[1]
        self.n_frames = n_frames
        self.weight_sl = weight_sl

        
        self.weights = torch.zeros(1, self.n_frames, self.height, self.width, self.weight_sl, self.weight_sl) # no params
        self.weights[:, :, :, :, (self.weight_sl-1)//2, (self.weight_sl-1)//2] += 30.0

    def forward(self, in_frames, **kwargs):
        return {'out': in_frames, 'positions': torch.zeros((in_frames.shape[0], in_frames.shape[1], in_frames.shape[3], in_frames.shape[4], 2)).to(in_frames.device), 'input': in_frames, 'weights': self.weights, 'exp_norm': torch.zeros(in_frames.shape).to(in_frames.device)}

class ConvFeaturePredictor(torch.nn.Module):
    def __init__(self, cfg, height, width, feat_dim):
        super(ConvFeaturePredictor, self).__init__()

        x_pos = torch.linspace(-1, 1, height)
        y_pos = torch.linspace(-1, 1, width)
        grid_x, grid_y = torch.meshgrid(x_pos, y_pos)
        xy = torch.cat((grid_x[None, None], grid_y[None, None]), dim=1)
        self.pos_emb = torch.zeros((1, 160, height, width))
        for i in range(16):
            self.pos_emb[:,4*i:4*i+2] = torch.sin(xy*(i+1)*3.1415)
            self.pos_emb[:,4*i+2:4*i+4] = torch.cos(xy*(i+1)*3.1415)
        for i in range(8):
            self.pos_emb[:,64+4*i:66+4*i] = torch.sin(xy*(2*i+16)*3.1415)
            self.pos_emb[:,66+4*i:68+4*i] = torch.cos(xy*(2*i+16)*3.1415)
        for i in range(8):
            self.pos_emb[:,96+4*i:98+4*i] = torch.sin(xy*(4*i+32)*3.1415)
            self.pos_emb[:,98+4*i:100+4*i] = torch.cos(xy*(4*i+32)*3.1415)
        for i in range(8):
            self.pos_emb[:,128+4*i:130+4*i] = torch.sin(xy*(8*i+64)*3.1415)
            self.pos_emb[:,130+4*i:132+4*i] = torch.cos(xy*(8*i+64)*3.1415)
        #self.pos_emb = torch.nn.Parameter(torch.randn(1, 15, height, width) * 0.0)

        self.conv1 = torch.nn.Conv2d(3 + self.pos_emb.shape[1], feat_dim, 123, padding=61)
        #self.conv2 = torch.nn.Conv2d(32, 32, 31, padding=15)
        #self.conv3 = torch.nn.Conv2d(32, 32, 31, padding=15)
        #self.conv4 = torch.nn.Conv2d(32, feat_dim, 31, padding=15)
        self.relu = torch.nn.ReLU()

        self.conv1.weight.data[:, 3:] = torch.zeros_like(self.conv1.weight.data[:, 3:])

    def forward(self, x):
        x = torch.cat((x, self.pos_emb.to(x.device)), dim=1)
        x = self.conv1(x)
        """
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        """

        return x

class OverfitSoftLearner(L.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg
        self.downsample_factor = cfg.model.downsample_factor
        self.filter_zoom = cfg.model.filter_zoom
        self.set_model = True
        self.three_frames = cfg.model.three_frames
        self.weight_sl = cfg.model.weight_sl
        #self.temp = cfg.model.temp
        self.temp = torch.nn.Parameter(torch.Tensor([cfg.model.temp]))
        self.bisided = cfg.model.bisided
        if self.bisided and self.cfg.model.cost_method == 'weights':
            raise ValueError('bisided flow optimization not compatible with weights')

        imsz_sm = [int(x) for x in cfg.dataset.imsz.split(",")]
        imsz_lg = [int(x) for x in cfg.dataset.imsz_super.split(",")]
        assert(imsz_sm[0] * self.downsample_factor == imsz_lg[0])
        assert(imsz_sm[1] * self.downsample_factor == imsz_lg[1])
        #self.temp_mtx = torch.nn.Parameter(torch.full((1, 1, imsz_sm[1], imsz_sm[0], 1, 1), cfg.model.temp))

        if self.three_frames:
            self.model = SoftMultiFramePredictor(cfg, 3, downsample_factor=self.downsample_factor, weight_sl=self.weight_sl, filter_zoom=self.filter_zoom, feat_dim=cfg.model.feat_dim) # 123 for num. 11
        else:
            self.model = SoftMultiFramePredictor(cfg, 2, downsample_factor=self.downsample_factor, weight_sl=self.weight_sl, filter_zoom=self.filter_zoom, feat_dim=cfg.model.feat_dim) # 123 for num. 11
        #self.low_model = SoftMultiFramePredictor(cfg, 2, downsample_factor=self.downsample_factor, weight_sl=3, filter_zoom=1)

        #self.tgt_model = SoftMultiFramePredictor(cfg, self.model.n_frames, weight_sl=3) # filter for target image
        self.tgt_model = IdentityPredictor(cfg, 1, weight_sl=3) # no filters for target image

        self.old_epe_full = None

    def configure_optimizers(self):
        #self.optimizers = torch.optim.Adam(list(self.model.parameters()) + list(self.tgt_model.parameters()), lr=self.cfg.training.lr, weight_decay=self.cfg.training.weight_decay)
        #self.optimizers = torch.optim.Adam(self.trainer.model.parameters(), lr=self.cfg.training.lr, weight_decay=self.cfg.training.weight_decay)
        self.optimizers = torch.optim.Adam(list(self.trainer.model.parameters()) + [self.temp], lr=self.cfg.training.lr, weight_decay=self.cfg.training.weight_decay)

        return self.optimizers

    def chunk(self, x):
        bsz = x.shape[0]
        x[:, 0, 0, 0] = x[:, 0, 0, 0] * 0.95  # not completely white so wandb doesn't bug out
        return list(torch.chunk(x, bsz))

    def loss(self, outputs, targets, occ_mask=None, image=None):
        if occ_mask is not None:
            mult_factor = (~occ_mask.repeat(1, 1, 3, 1, 1)).float()
        else:
            mult_factor = torch.ones_like(targets["out"])

        """
        if self.three_frames:
            loss = criterion_three_frames(outputs["out"] * mult_factor, targets["out"] * mult_factor)
            #loss = criterion(outputs["out"] * mult_factor, targets["out"] * mult_factor)
        else:
            loss = criterion(outputs["out"] * mult_factor, targets["out"] * mult_factor)
        #loss, small_src = distribution_photometric(outputs['weights'], outputs['input'], targets['out'], self.model.weight_sl, self.model.downsample_factor, self.model.filter_zoom)

        assert(self.model.filter_zoom == 1)
        """
        loss, _ = normal_minperblob(outputs['weights'], outputs['input'], targets['out'], self.model.weight_sl, self.model.downsample_factor, self.model.filter_zoom, flatness=self.cfg.model.charbonnier_flatness, use_min=(self.global_step > 299))

        #input(small_src.shape)
        #small_outputs = self.low_model(torch.zeros_like(outputs['input']))
        #loss2, _ = distribution_photometric(small_outputs['weights'], small_src, targets['out'], self.low_model.weight_sl, self.low_model.downsample_factor, self.low_model.filter_zoom)
        #loss += loss2

        if self.three_frames:
            temp_smooth_reg = temporal_smoothness_loss(outputs['positions'])
            #temp_smooth_reg = full_temporal_smoothness_loss(outputs['weights'])
            loss += self.cfg.model.temp_smoothness * temp_smooth_reg

        if occ_mask is not None:
            occ_mask = occ_mask.float()
        spat_smooth_reg = spatial_smoothness_loss(outputs['weights'], image=image, occ_mask=occ_mask)
        loss += spat_smooth_reg * self.cfg.model.smoothness * (self.model.true_sl ** 2) # finally weighing this #2.8e4 #* (61/31) * (61/31) # for sintel

        #spat_smooth_reg = spatial_smoothness_loss(targets['weights'], image=targets["input"])
        #loss += spat_smooth_reg * 5e-0 # 100x smaller

        spat_smooth_reg = position_spat_smoothness(outputs['positions'])
        loss += spat_smooth_reg * self.cfg.model.pos_smoothness #2e-3

        norm_reg = outputs['exp_norm'].mean()
        loss += norm_reg * self.cfg.model.norm

        #norm_reg = targets['exp_norm'].mean()
        #loss += norm_reg * 1e-1

        entropy_reg = entropy_loss(outputs['weights'])
        loss += entropy_reg * self.cfg.model.entropy

        #entropy_reg = entropy_loss(targets['weights'])
        #loss += entropy_reg * 1e-1

        #self_reg = torch.nn.functional.mse_loss(targets['input'], targets['out'])
        #loss += self_reg * 1e-1
        """
        raft = raft_large(weights='C_T_V2')
        raft.to(targets_up.device)
        flow = raft(targets_up, outputs["input"][0])
        flow[-1] = torchvision.transforms.functional.resize(flow[-1], (flow[-1].shape[2]//self.downsample_factor, flow[-1].shape[3]//self.downsample_factor))
        loss = torch.mean(torch.abs(torch.permute(outputs['positions'][0], (0, 3, 1, 2)) - flow[-1]))
        """
        #print(f"Final {loss}")

        return loss

    def check_filter_size(self, frame2, frame3, gt_flow):
        flow_max = torch.max(torch.abs(gt_flow))
        if flow_max == 0: # no gt flow
            raft = raft_large(weights='C_T_V2')
            raft.to(frame2.device)

            mem_limit_scale = 1 / max(frame2.shape[2] / 1024, frame2.shape[3] / 1024, 1) # scale down for memory limits
            frame2_round = torchvision.transforms.functional.resize(frame2, (8 * math.ceil(frame2.shape[2] / 8 * mem_limit_scale), 8 * math.ceil(frame2.shape[3] / 8 * mem_limit_scale)))
            frame3_round = torchvision.transforms.functional.resize(frame3, (8 * math.ceil(frame3.shape[2] / 8 * mem_limit_scale), 8 * math.ceil(frame3.shape[3] / 8 * mem_limit_scale)))
            gt_flow = raft(frame2_round, frame3_round)[-1]

            flow_max = torch.max(torch.abs(gt_flow)) / (mem_limit_scale * self.downsample_factor)
        if flow_max > self.model.max_size:
            raise ValueError(f'GT flow is too large--max flow is {flow_max} but max possible is {self.model.max_size}')
        else:
            return flow_max

    def training_step(self, batch, batch_idx):
        frame1, frame2, frame3, gt_flow, frame1_up, frame2_up, frame3_up, gt_flow_orig = batch
        #self.check_filter_size(frame2_up, frame3_up, gt_flow)

        #outputs = self.model(frame3_up[:, None], frame2[:, None], temp=self.temp, in_down=frame3[:, None])
        #if self.bisided:
        #    outputs_2 = self.model(frame2_up[:, None], frame3[:, None], temp=self.temp, src_idx=1, tgt_idx=0, in_down=frame2[:, None])
        #elif self.three_frames:
        #    outputs_2 = self.model(frame1_up[:, None], frame2[:, None], temp=self.temp, src_idx=1, tgt_idx=0, in_down=frame2[:, None])
        if self.bisided:
            assert(not self.three_frames)
            outputs = self.model(torch.cat((frame3_up[:, None], frame2_up[:, None]), dim=1), 
                                 torch.cat((frame2[:, None], frame3[:, None]), dim=1), 
                                 temp=self.temp, 
                                 in_down=torch.cat((frame3[:, None], frame2[:, None]), dim=1),
                                 src_idx=[0,1],
                                 tgt_idx=[1,0])
            target = self.tgt_model(torch.cat((frame2[:, None], frame3[:, None]), dim=1))
        elif self.three_frames:
            outputs = self.model(torch.cat((frame3_up[:, None], frame1_up[:, None]), dim=1), 
                                 torch.cat((frame2[:, None], frame2[:, None]), dim=1), 
                                 temp=self.temp, 
                                 in_down=torch.cat((frame3[:, None], frame1[:, None]), dim=1),
                                 src_idx=[0,2],
                                 tgt_idx=[1,1])
            target = self.tgt_model(torch.cat((frame2[:, None], frame2[:, None]), dim=1))
        else:
            outputs = self.model(frame3_up[:, None], frame2[:, None], temp=self.temp, in_down=frame3[:, None])
            target = self.tgt_model(frame2[:, None])

        """
        target = self.tgt_model(frame2[:, None])
        if self.bisided:
            target_2 = self.tgt_model(frame3[:, None])
        """

        fwd_flows = [(outputs["positions"] - target["positions"])[:, -1].permute(0, 3, 1, 2).float()]

        # this part only works with 2 frames
        if self.three_frames:
            """
            swap_flows = torch.flip(outputs["positions"] - target["positions"], dims=[1]).permute(0, 1, 4, 2, 3).float()
            swap_flows = torch.reshape(swap_flows, (swap_flows.shape[0] * 2, *swap_flows.shape[2:]))
            occ_mask = softsplat(torch.zeros_like(-swap_flows), -swap_flows, None, strMode="sum")[:, -1, None] # self occ
            occ_mask = torch.reshape(occ_mask, (occ_mask.shape[0] // 2, 2, *occ_mask.shape[1:]))
            occ_mask = (occ_mask > 1.2).detach()
            """
            occ_mask = None
        else:
            occ_mask = None

        """
        loss = self.loss(outputs, target, occ_mask=occ_mask, image=frame2[:, None])
        if self.bisided: # TODO 7/5: make 3 frames work again
            loss += self.loss(outputs_2, target_2, occ_mask=occ_mask, image=frame3[:, None])
        """
        loss = 2 * self.loss(outputs, target, occ_mask=occ_mask, image=torch.cat((frame2[:, None], frame3[:, None]), dim=1))
        fullres_fwd_flow = fwd_flows[-1] # correction deleted for now
        self.log_dict({
            "train/loss": loss,
            "train/flow_fwd_min": torch.min(fullres_fwd_flow),
            "train/flow_fwd_max": torch.max(fullres_fwd_flow),
            "train/flow_fwd_mean": torch.mean(fullres_fwd_flow),
            "train/flow_fwd_std": torch.mean(torch.std(fullres_fwd_flow, dim=0)),
            "train/flow_gt_min": torch.min(gt_flow),
            "train/flow_gt_max": torch.max(gt_flow),
            "train/flow_gt_mean": torch.mean(gt_flow),
            "train/flow_gt_std": torch.mean(torch.std(gt_flow, dim=0))
        })

        self.log('loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        frame1, frame2, frame3, gt_flow, frame1_up, frame2_up, frame3_up, gt_flow_orig = batch
        if self.three_frames:
            flow_max = max(self.check_filter_size(frame2_up, frame3_up, gt_flow), self.check_filter_size(frame2_up, frame1_up, gt_flow))
        else:
            flow_max = self.check_filter_size(frame2_up, frame3_up, gt_flow)

        raft = raft_large(weights='C_T_V2')
        raft.to(frame2.device)
        smurf = raft_smurf(checkpoint="smurf_sintel.pt")
        smurf.to(frame2.device)

        if not self.set_model:
            #self.model.set(1, flow_to_filter(gt_flow, self.model.weight_sl))
            #self.set_model = True
            raise NotImplementedError # not compatible rn

        with torch.no_grad():
            if self.three_frames:
                outputs = self.model(torch.cat((frame3_up[:, None], frame1_up[:, None]), dim=1), 
                                     torch.cat((frame2[:, None], frame2[:, None]), dim=1), 
                                     temp=self.temp, 
                                     in_down=torch.cat((frame3[:, None], frame1[:, None]), dim=1),
                                     src_idx=[0,2],
                                     tgt_idx=[1,1])
                outputs_100 = self.model(torch.cat((frame3_up[:, None], frame1_up[:, None]), dim=1), 
                                     torch.cat((frame2[:, None], frame2[:, None]), dim=1), 
                                     temp=self.temp * 100, 
                                     in_down=torch.cat((frame3[:, None], frame1[:, None]), dim=1),
                                     src_idx=[0,2],
                                     tgt_idx=[1,1])
            else:
                outputs = self.model(frame3_up[:, None], frame2[:, None], temp=self.temp, in_down=frame3[:, None])
                outputs_2 = self.model(frame2_up[:, None], frame3[:, None], temp=self.temp, src_idx=1, tgt_idx=0, in_down=frame2[:, None])
                outputs_100 = self.model(frame3_up[:, None], frame2[:, None], temp=self.temp * 100, in_down=frame3[:, None])

            target_outputs = self.tgt_model(frame2[:, None])
            target_outputs_2 = self.tgt_model(frame3[:, None])
            fwd_flows = [(outputs["positions"] - target_outputs["positions"])[:, 0].permute(0, 3, 1, 2).float()]
            fwd_flows_100 = [(outputs_100["positions"] - target_outputs["positions"])[:, 0].permute(0, 3, 1, 2).float()]
            if self.three_frames:
                bwd_flows = [(outputs["positions"] - target_outputs["positions"])[:, 1].permute(0, 3, 1, 2).float()]
            else:
                bwd_flows = [torch.zeros_like(fwd_flows[-1]).float()]
            tgt_flows = [target_outputs["positions"][:, 0].permute(0, 3, 1, 2).float()]
            print(f'Max flow: {flow_max}, max pred flow: {torch.max(torch.abs(fwd_flows[-1]))}')

            loss = self.loss(outputs, target_outputs, occ_mask=None, image=frame2[:, None])
            if self.bisided:
                loss += self.loss(outputs_2, target_outputs_2, occ_mask=None, image=frame3[:, None])

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

            mem_limit_scale = 1 / max(frame2_up.shape[2] / 1024, frame2_up.shape[3] / 1024, 1) # scale down for memory limits
            frame2_round = torchvision.transforms.functional.resize(frame2, (8 * math.ceil(frame2.shape[2] / 8), 8 * math.ceil(frame2.shape[3] / 8)))
            frame3_round = torchvision.transforms.functional.resize(frame3, (8 * math.ceil(frame3.shape[2] / 8), 8 * math.ceil(frame3.shape[3] / 8)))
            frame2_up_round = torchvision.transforms.functional.resize(frame2_up, (8 * math.ceil(frame2_up.shape[2] / 8 * mem_limit_scale), 8 * math.ceil(frame2_up.shape[3] / 8 * mem_limit_scale)))
            frame3_up_round = torchvision.transforms.functional.resize(frame3_up, (8 * math.ceil(frame3_up.shape[2] / 8 * mem_limit_scale), 8 * math.ceil(frame3_up.shape[3] / 8 * mem_limit_scale)))
            
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

            if torch.any(torch.isnan(raft_pred)):
                raft_pred = torch.where(torch.isnan(raft_pred), 0.0, raft_pred)
            if torch.any(torch.isnan(raft_pred_full)):
                raft_pred_full = torch.where(torch.isnan(raft_pred_full), 0.0, raft_pred_full)
            if torch.any(torch.isnan(smurf_pred)):
                smurf_pred = torch.where(torch.isnan(smurf_pred), 0.0, smurf_pred)
            if torch.any(torch.isnan(smurf_pred_full)):
                smurf_pred_full = torch.where(torch.isnan(smurf_pred_full), 0.0, smurf_pred_full)

            # EPE at resolution
            raft_epe = compute_epe(gt_flow_orig, raft_pred)
            smurf_epe = compute_epe(gt_flow_orig, smurf_pred)

            if torch.max(torch.abs(gt_flow_orig)) == 0.0:
                gt_flow_orig = raft_pred_full
                gt_flow = torch.nn.functional.interpolate(raft_pred_full, size=(frame2.shape[2], frame2.shape[3]))
                gt_flow = gt_flow / self.downsample_factor
            epe_full = compute_epe(gt_flow_orig, fwd_flows[-1])
            mode_epe_full = compute_epe(gt_flow_orig, fwd_flows_100[-1])
            raw_epe_full = compute_epe(gt_flow_orig, fwd_flows[-1] + tgt_flows[-1]) # original epe
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
            """
            occ_mask = softsplat(torch.zeros_like(gt_flow_orig), gt_flow_orig, None, strMode="sum")[:, -1, None]
            occ_mask = occ_mask > 1.5
            masked_gt_flow_orig = torch.where(occ_mask.repeat(1, 2, 1, 1), torch.full_like(gt_flow_orig, float('nan')), gt_flow_orig)
            occ_epe = compute_epe(masked_gt_flow_orig, fwd_flows[-1] + tgt_flows[-1])

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
                "pos_emb_avg": 0.0 if self.cfg.model.cost_method != "nn" else torch.mean(torch.abs(self.model.src_nn.pos_emb)),
            }, sync_dist=True)
            print(epe_full)
            #"val/var_mean": torch.mean(outputs["var"])

            fwd_flow = fwd_flows[-1]
            fwd_flow_100 = fwd_flows_100[-1]
            bwd_flow = bwd_flows[-1]
            tgt_flow = tgt_flows[-1]
            target = target_outputs["out"][:, 0]

            warped_imgs_flow = warp(frame2, fwd_flow)
            warped_imgs_fwd = outputs["out"][:, -1]

            fwd_flows = torchvision.utils.flow_to_image(fwd_flow) / 255 * torch.max(frame2)
            fwd_flows_100 = torchvision.utils.flow_to_image(fwd_flow_100) / 255 * torch.max(frame2)
            bwd_flows = torchvision.utils.flow_to_image(bwd_flow) / 255 * torch.max(frame2)
            gt_flows = torchvision.utils.flow_to_image(gt_flow) / 255 * torch.max(frame2)
            diff_flows = torchvision.utils.flow_to_image(fwd_flow - gt_flow) / 255 * torch.max(frame2)
            gradf_flows = torchvision.utils.flow_to_image(grad_fwd_flow.float()) / 255 * torch.max(frame2)
            gradb_flows = torchvision.utils.flow_to_image(grad_bwd_flow.float()) / 255 * torch.max(frame2)
            tgt_flows = torchvision.utils.flow_to_image(tgt_flow) / 255 * torch.max(frame2)
            raft_pred = torchvision.utils.flow_to_image(raft_pred) / 255 * torch.max(frame2)
            raft_pred_full = torchvision.utils.flow_to_image(raft_pred_full) / 255 * torch.max(frame2)
            smurf_pred = torchvision.utils.flow_to_image(smurf_pred) / 255 * torch.max(frame2)
            smurf_pred_full = torchvision.utils.flow_to_image(smurf_pred_full) / 255 * torch.max(frame2)

            U = torch.linspace(-1, 1, 100)
            V = torch.linspace(-1, 1, 100)
            X, Y = torch.meshgrid(U, V)
            wheel_flow = torch.stack((X, Y), dim=0)[None].to(frame2.device)
            wheel_flow = torchvision.utils.flow_to_image(wheel_flow) / 255 * torch.max(frame2)

            filters = filter_to_image(outputs["weights"][:, -1], downsample_factor=32)
            target_filters = filter_to_image(target_outputs["weights"][:, -1], downsample_factor=16)

            combined_frames = torch.cat((frame1, frame2, frame3), dim=3)
            target = torch.cat((target, frame2, target-frame2), dim=3)
            fwd_flow = torch.cat((frame2, frame3, fwd_flows), dim=3)
            bwd_flow = torch.cat((frame1, frame2, bwd_flows), dim=3)
            tgt_flow = torch.cat((frame3, tgt_flows), dim=3)
            fwd_warped = torch.cat((frame2, frame3, warped_imgs_fwd, 0.5 + 0.5 * (warped_imgs_fwd - frame2)), dim=3)
            fwd_warped_flow = torch.cat((frame2, frame3, warped_imgs_flow, 0.5 + 0.5 * (warped_imgs_flow - frame2)), dim=3)
            gt_fwd = torch.cat((gt_flows, fwd_flows), dim=3)
            gt_fwd_100 = torch.cat((gt_flows, fwd_flows_100), dim=3)
            gt_fwd_raft = torch.cat((torchvision.transforms.functional.resize(gt_flows, (raft_pred.shape[2], raft_pred.shape[3])), raft_pred), dim=3)
            gt_fwd_raft_full = torch.cat((torchvision.transforms.functional.resize(gt_flows, (raft_pred_full.shape[2], raft_pred_full.shape[3])), raft_pred_full), dim=3)
            gt_fwd_smurf = torch.cat((torchvision.transforms.functional.resize(gt_flows, (smurf_pred.shape[2], smurf_pred.shape[3])), smurf_pred), dim=3)
            gt_fwd_smurf_full = torch.cat((torchvision.transforms.functional.resize(gt_flows, (smurf_pred_full.shape[2], smurf_pred_full.shape[3])), smurf_pred_full), dim=3)
            occ_mask_vis = torch.cat((torchvision.transforms.functional.resize(gt_flows, (occ_mask.shape[2], occ_mask.shape[3])), occ_mask.repeat(1, 3, 1, 1).float()), dim=3)

            # Soft flow maps f1, f3 to f2
            self.logger.log_image(key='combined_frames', images=self.chunk(combined_frames), step=self.global_step) # f1, f2, f3 combined
            self.logger.log_image(key='fwd_flow', images=self.chunk(fwd_flow), step=self.global_step) # f2, f3, then predicted forward flow from soft flows
            self.logger.log_image(key='bwd_flow', images=self.chunk(bwd_flow), step=self.global_step) # not used
            self.logger.log_image(key='tgt_flow', images=self.chunk(tgt_flow), step=self.global_step) # flow from the target filters
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
            self.logger.log_image(key='target_filters', images=self.chunk(target_filters), step=self.global_step) # visualizing the filters inside the target soft flow
            self.logger.log_image(key='exp_norm', images=self.chunk(outputs['exp_norm'][:, -1, None].repeat(1,3,1,1)), step=self.global_step) # grayscale image of the expected norm of the flow
            #self.logger.log_image(key='var', images=self.chunk(outputs['var'][:, -1, None].repeat(1,3,1,1)), step=self.global_step) # grayscale image of the expected norm of the flow
            self.logger.log_image(key='occ_mask', images=self.chunk(occ_mask_vis), step=self.global_step) # grayscale image of the expected norm of the flow
            self.logger.log_image(key='colorwheel', images=self.chunk(wheel_flow), step=self.global_step) # grayscale image of the expected norm of the flow
            #self.logger.log_image(key='temp_img', images=self.chunk(torch.squeeze(self.temp)[None, None].repeat(1, 3, 1, 1)), step=self.global_step) # grayscale image of the expected norm of the flow

            if self.cfg.model.cost_method in ['feats', 'nn']:
                if self.cfg.model.cost_method == 'feats':
                    src_feats = self.model.feats[:, 0, None] / torch.linalg.norm(self.model.feats[:, 0, None], dim=2, keepdim=True)
                    tgt_feats = self.model.feats[:, 1, None] / torch.linalg.norm(self.model.feats[:, 1, None], dim=2, keepdim=True)
                else:
                    src_feats = self.model.nn(frame3)[:, None]
                    tgt_feats = self.model.nn(frame2)[:, None]
                    src_feats = src_feats / torch.linalg.norm(src_feats, dim=2, keepdim=True)
                    tgt_feats = tgt_feats / torch.linalg.norm(tgt_feats, dim=2, keepdim=True)
                    src_pos_emb = self.model.nn.pos_emb[:, None]
                    tgt_pos_emb = self.model.nn.pos_emb[:, None]
                    src_pos_emb = src_pos_emb - torch.min(src_pos_emb, dim=2, keepdim=True)[0]
                    src_pos_emb = src_pos_emb / (torch.max(src_pos_emb, dim=2, keepdim=True)[0] + 1e-10)
                    tgt_pos_emb = tgt_pos_emb - torch.min(tgt_pos_emb, dim=2, keepdim=True)[0]
                    tgt_pos_emb = tgt_pos_emb / (torch.max(tgt_pos_emb, dim=2, keepdim=True)[0] + 1e-10)
                #src_atan = torchvision.utils.flow_to_image(src_feats[0, :])
                #tgt_atan = torchvision.utils.flow_to_image(tgt_feats[0, :])
                src_feats = (torch.permute(src_feats[0], (1, 0, 2, 3)) * 0.5 + 0.5) * 255
                tgt_feats = (torch.permute(tgt_feats[0], (1, 0, 2, 3)) * 0.5 + 0.5) * 255
                if self.cfg.model.cost_method == "nn":
                    src_pos_emb = torch.permute(src_pos_emb[0], (1, 0, 2, 3)) * 255
                    tgt_pos_emb = torch.permute(tgt_pos_emb[0], (1, 0, 2, 3)) * 255
                else:
                    src_pos_emb, tgt_pos_emb = None, None
                self.logger.log_image(key='src_feats', images=self.chunk(src_feats), step=self.global_step)
                self.logger.log_image(key='tgt_feats', images=self.chunk(tgt_feats), step=self.global_step)
                if src_pos_emb is not None and src_pos_emb.shape[0] <= 32: # probably a learned embedding
                    self.logger.log_image(key='src_pos_emb', images=self.chunk(src_pos_emb), step=self.global_step)
                    self.logger.log_image(key='tgt_pos_emb', images=self.chunk(tgt_pos_emb), step=self.global_step)
                #self.logger.log_image(key='src_feats_atan', images=self.chunk(src_atan), step=self.global_step)
                #self.logger.log_image(key='tgt_feats_atan', images=self.chunk(tgt_atan), step=self.global_step)
                
                if self.cfg.model.temp_scheduling:
                    # Temperature increases when EPE no longer increases.
                    if self.old_epe_full != None and self.old_epe_full - 0.01 < epe_full:
                        print(f"Old temp: {self.temp}")
                        self.temp *= 1.1
                        print(f"New temp: {self.temp}")
                self.old_epe_full = epe_full

def compute_epe(gt, pred):
    _, _, h_pred, w_pred = pred.size()
    bs, nc, h_gt, w_gt = gt.size()

    u_gt, v_gt = gt[:,0,:,:], gt[:,1,:,:]
    pred = torch.nn.functional.upsample(pred, size=(h_gt, w_gt), mode='bilinear')
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
