import pytorch_lightning as pl
from omegaconf import DictConfig

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.optical_flow import raft_large

from soft_losses import *
from soft_utils import flow_from_weights, filter_to_image, flow_to_filter

class SoftMultiFramePredictor(torch.nn.Module):
    def __init__(self, cfg, n_frames, weight_sl=31, downsample_factor=1, init='all'):
        super(SoftMultiFramePredictor, self).__init__()
        image_size = [int(x) for x in cfg.dataset.imsz.split(",")]
        image_size = [image_size[1], image_size[0]]

        self.height = image_size[0]
        self.width = image_size[1]
        self.n_frames = n_frames
        self.weight_sl = weight_sl
        self.downsample_factor = downsample_factor
        self.weights = torch.nn.Parameter(torch.ones(1, self.n_frames, self.height, self.width, self.weight_sl + self.downsample_factor - 1, self.weight_sl + self.downsample_factor - 1)*1e-2) 
        if init == 'zero':
            assert(downsample_factor == 1)
            self.weights.data[:, :, :, :, (self.weight_sl-1)//2, (self.weight_sl-1)//2] += 10.0

        t = self.weight_sl//2 + (self.downsample_factor - 1)/2
        x_positions = torch.linspace(-t, t, self.weight_sl + self.downsample_factor - 1)
        y_positions = torch.linspace(-t, t, self.weight_sl + self.downsample_factor - 1)
        grid_x, grid_y = torch.meshgrid(x_positions, y_positions)
        self.grid = torch.stack((grid_x, grid_y), dim=-1) 

    def forward(self, in_frames, weights=None):
        if weights is not None:
            weights = weights.flatten(start_dim=-2)
        else:
            weights = self.weights.flatten(start_dim=-2)
        weights = F.softmax(weights, dim=-1).view_as(self.weights)

        weights = weights.to(in_frames.device)
        grid = self.grid.to(in_frames.device)

        t = self.weight_sl//2 # smaller t for padding
        in_shape = in_frames.shape
        in_frames_reshaped = torch.reshape(in_frames, (in_frames.shape[0] * in_frames.shape[1], in_frames.shape[2], in_frames.shape[3], in_frames.shape[4]))
        x_padded = F.pad(in_frames_reshaped, (t, t, t, t), mode='replicate', value=0)
        x_padded = torch.reshape(x_padded, (in_shape[0], in_shape[1], *x_padded.shape[1:]))
        x_padded = x_padded.unfold(3, self.weight_sl + self.downsample_factor - 1, self.downsample_factor).unfold(4, self.weight_sl + self.downsample_factor - 1, self.downsample_factor)  # extract 30x30 patches around each pixel
        # should be perfect downsampling now

        # For each pixel in the input frame, compute the weighted sum of its neighborhood
        out = torch.einsum('bnijkl, bncijkl->bncij', weights, x_padded)
        expected_positions = torch.einsum('bnijkl,klm->bnijm', weights, grid) / self.downsample_factor # Shape: [height, width, 2]
        expected_norm = torch.einsum('bnijkl,kl->bnij', weights, grid.norm(dim=-1)) / self.downsample_factor  # Shape: [height, width, 2]
        expected_positions = expected_positions.flip(-1)

        return {'out':out, 'positions':expected_positions, 'input':in_frames, 'weights':weights, 'exp_norm':expected_norm}

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

class OverfitSoftLearner(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg
        self.model = SoftMultiFramePredictor(cfg, 2, downsample_factor=4, weight_sl=31)
        self.set_model = True

        #self.tgt_model = SoftMultiFramePredictor(cfg, 2, weight_sl=3, init='zero') # filter for target image
        self.tgt_model = IdentityPredictor(cfg, 2, weight_sl=3) # no filters for target image

    def configure_optimizers(self):
        self.optimizers = torch.optim.Adam(list(self.model.parameters()) + list(self.tgt_model.parameters()), lr=self.cfg.training.lr, weight_decay=self.cfg.training.weight_decay)

        return self.optimizers

    def chunk(self, x):
        bsz = x.shape[0]
        x[:, 0, 0, 0] = x[:, 0, 0, 0] * 0.95  # not completely white so wandb doesn't bug out
        return list(torch.chunk(x, bsz))

    def loss(self, outputs, targets):
        loss = criterion(outputs["out"], targets["out"])

        temp_smooth_reg = temporal_smoothness_loss(outputs['positions'])
        # loss += temp_smooth_reg

        spat_smooth_reg = spatial_smoothness_loss(outputs['weights'])
        loss += spat_smooth_reg * 5e2 #* (61/31) * (61/31) # for sintel

        spat_smooth_reg = spatial_smoothness_loss(targets['weights'])
        loss += spat_smooth_reg * 5e-0 # 100x smaller

        #spat_smooth_reg = position_spat_smoothness(outputs['positions'])
        #loss += spat_smooth_reg * 0.002

        norm_reg = outputs['exp_norm'].mean()
        loss += norm_reg * 1e-3

        #norm_reg = targets['exp_norm'].mean()
        #loss += norm_reg * 1e-1

        entropy_reg = entropy_loss(outputs['weights'])
        # loss += entropy_reg

        entropy_reg = entropy_loss(targets['weights'])
        #loss += entropy_reg * 1e-1

        self_reg = torch.nn.functional.mse_loss(targets['input'], targets['out'])
        loss += self_reg * 1e-1

        return loss

    def training_step(self, batch, batch_idx):
        frame1, frame2, frame3, gt_flow, frame1_up, frame2_up, frame3_up, gt_flow_orig = batch

        outputs = self.model(torch.stack((frame1_up, frame3_up), dim=1))
        fwd_flows = [outputs["positions"]]

        target = self.tgt_model(torch.stack((frame2, frame2), dim=1))

        loss = self.loss(outputs, target)
        fullres_fwd_flow = fwd_flows[-1] - target["positions"]
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
        raft = raft_large(weights='C_T_V2')
        raft.to(frame2.device)

        if not self.set_model:
            self.model.set(1, flow_to_filter(gt_flow, self.model.weight_sl))
            self.set_model = True

        with torch.no_grad():
            outputs = self.model(torch.stack((frame1_up, frame3_up), dim=1))
            target_outputs = self.tgt_model(torch.stack((frame2, frame2), dim=1))
            fwd_flows = [(outputs["positions"] - target_outputs["positions"])[:, 1].permute(0, 3, 1, 2)]
            bwd_flows = [torch.zeros_like(fwd_flows[-1])]
            tgt_flows = [target_outputs["positions"][:, 1].permute(0, 3, 1, 2)]

            loss = self.loss(outputs, target_outputs)
            with torch.set_grad_enabled(True):
                weights = self.model.weights.detach().clone()
                weights.requires_grad_(True)
                outputs_grad = self.model(torch.stack((frame1_up, frame3_up), dim=1), weights=weights)

                loss = self.loss(outputs_grad, target_outputs)
                loss.backward()

                grad_fwd_flow = -weights.grad[:, 1].clone() # negative; reflects the drcn of grad descent
                grad_fwd_flow = flow_from_weights(grad_fwd_flow)
                try:
                    grad_bwd_flow = -weights.grad[:, 0].clone() 
                    grad_bwd_flow = flow_from_weights(grad_bwd_flow)
                except:
                    grad_bwd_flow = torch.zeros_like(bwd_flows[-1])

                weights.grad = None
                weights.requires_grad_(False)

            raft_pred = raft(frame2, frame3)[-1]
            raft_pred_full = raft(frame2_up, frame3_up)[-1]

            # EPE at resolution
            raft_epe = compute_epe(gt_flow_orig, raft_pred)
            epe_full = compute_epe(gt_flow_orig, fwd_flows[-1])
            raw_epe_full = compute_epe(gt_flow_orig, fwd_flows[-1] + tgt_flows[-1]) # original epe
            raft_epe_full = compute_epe(gt_flow_orig, raft_pred_full)

            self.log_dict({
                "val/loss": loss, # training loss
                "val/raft_epe": raft_epe, # EPE between (1) RAFT with downscaled f2, f3 as input (2) full resolution flow
                "val/epe_full": epe_full, # EPE between (1) flow from the soft flow overfitting on downscaled inputs (2) full resolution flow
                "val/raw_epe_full": raw_epe_full, # previous EPE but w/o target filter adjustment (same if no target filters)
                "val/raft_epe_full": raft_epe_full, # EPE between (1) RAFT with full resolution f2, f3 as input (2) full resolution flow
            }, sync_dist=True)

            fwd_flow = fwd_flows[-1]
            bwd_flow = bwd_flows[-1]
            tgt_flow = tgt_flows[-1]
            target = target_outputs["out"][:, 0]

            warped_imgs_flow = warp(frame2, fwd_flow)
            warped_imgs_fwd = outputs["out"][:, 1]

            fwd_flows = torchvision.utils.flow_to_image(fwd_flow) / 255 * torch.max(frame2)
            bwd_flows = torchvision.utils.flow_to_image(bwd_flow) / 255 * torch.max(frame2)
            gt_flows = torchvision.utils.flow_to_image(gt_flow) / 255 * torch.max(frame2)
            diff_flows = torchvision.utils.flow_to_image(fwd_flow - gt_flow) / 255 * torch.max(frame2)
            gradf_flows = torchvision.utils.flow_to_image(grad_fwd_flow) / 255 * torch.max(frame2)
            gradb_flows = torchvision.utils.flow_to_image(grad_bwd_flow) / 255 * torch.max(frame2)
            tgt_flows = torchvision.utils.flow_to_image(tgt_flow) / 255 * torch.max(frame2)
            raft_pred = torchvision.utils.flow_to_image(raft_pred) / 255 * torch.max(frame2)
            raft_pred_full = torchvision.utils.flow_to_image(raft_pred_full) / 255 * torch.max(frame2)

            filters = filter_to_image(outputs["weights"][:, 1], downsample_factor=32)
            target_filters = filter_to_image(target_outputs["weights"][:, 1], downsample_factor=16)

            combined_frames = torch.cat((frame1, frame2, frame3), dim=3)
            target = torch.cat((target, frame2, target-frame2), dim=3)
            fwd_flow = torch.cat((frame2, frame3, fwd_flows), dim=3)
            bwd_flow = torch.cat((frame1, frame2, bwd_flows), dim=3)
            tgt_flow = torch.cat((frame3, tgt_flows), dim=3)
            fwd_warped = torch.cat((frame2, frame3, warped_imgs_fwd, 0.5 + 0.5 * (warped_imgs_fwd - frame2)), dim=3)
            fwd_warped_flow = torch.cat((frame2, frame3, warped_imgs_flow, 0.5 + 0.5 * (warped_imgs_flow - frame2)), dim=3)
            gt_fwd = torch.cat((gt_flows, fwd_flows), dim=3)
            gt_fwd_raft = torch.cat((gt_flows, raft_pred), dim=3)
            gt_fwd_raft_full = torch.cat((torchvision.transforms.functional.resize(gt_flows, (raft_pred_full.shape[2], raft_pred_full.shape[3])), raft_pred_full), dim=3)

            # Soft flow maps f1, f3 to f2
            self.logger.log_image(key='combined_frames', images=self.chunk(combined_frames), step=self.global_step) # f1, f2, f3 combined
            self.logger.log_image(key='fwd_flow', images=self.chunk(fwd_flow), step=self.global_step) # f2, f3, then predicted forward flow from soft flows
            self.logger.log_image(key='bwd_flow', images=self.chunk(bwd_flow), step=self.global_step) # not used
            self.logger.log_image(key='tgt_flow', images=self.chunk(tgt_flow), step=self.global_step) # flow from the target filters
            self.logger.log_image(key='fwd_warped', images=self.chunk(fwd_warped), step=self.global_step) # f2, f3, then the output of the soft flow (i.e. f3 mapped to f2)
            self.logger.log_image(key='fwd_warped_flow', images=self.chunk(fwd_warped_flow), step=self.global_step) # f2, f3, then f3 backward warped with the predicted flow onto f2
            self.logger.log_image(key='target', images=self.chunk(target), step=self.global_step) # f2 after target filter, f2, and the difference between the two
            self.logger.log_image(key='gt_fwd_flow', images=self.chunk(gt_fwd), step=self.global_step) # gt forward flow (downscaled), predicted flow
            self.logger.log_image(key='gt_raft', images=self.chunk(gt_fwd_raft), step=self.global_step) # gt forward flow (downscaled), RAFT output with downscaled inputs
            self.logger.log_image(key='gt_raft_full', images=self.chunk(gt_fwd_raft_full), step=self.global_step) # gt forward flow (full resolution), RAFT output with full resolution inputs
            self.logger.log_image(key='diff_flow', images=self.chunk(diff_flows), step=self.global_step) # take the difference between predicted flow and gt, and visualize (this is another flow)
            self.logger.log_image(key='grad_fwd_flow', images=self.chunk(gradf_flows), step=self.global_step) # gradient of the forward flow
            self.logger.log_image(key='grad_bwd_flow', images=self.chunk(gradb_flows), step=self.global_step) # not used
            self.logger.log_image(key='filters', images=self.chunk(filters), step=self.global_step) # visualizing the filters inside the soft flow (1 filter shown per 32x32 pixels)
            self.logger.log_image(key='target_filters', images=self.chunk(target_filters), step=self.global_step) # visualizing the filters inside the target soft flow
            self.logger.log_image(key='exp_norm', images=self.chunk(outputs['exp_norm'][:, 1, None].repeat(1,3,1,1)), step=self.global_step) # grayscale image of the expected norm of the flow


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

    return epe.mean()

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
