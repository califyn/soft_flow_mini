import torch, torchvision
from torchvision.utils import flow_to_image

from omegaconf import OmegaConf
import math
from tqdm import tqdm
from copy import deepcopy

from scipy.interpolate import LinearNDInterpolator

from data.superres import Batch
from data.sintel_superres import SintelSuperResDataset
from data.spring_superres import SpringSuperResDataset
from data.kitti_superres import KITTISuperResDataset

from RAFT.core.raft import RAFT
import RAFT.core.datasets as datasets
from RAFT.core.utils.utils import InputPadder
from smurf import raft_smurf
from torchvision.models.optical_flow import raft_small, raft_large

import numpy as np
import traceback

import os, sys
import lightning as L
from lightning.pytorch.loggers.wandb import WandbLogger
from src.learner import OverfitSoftLearner
import wandb, logging
import hashlib

def compute_epe(gt, pred, scale_to=None, mask=None):
    _, _, h_pred, w_pred = pred.size()
    bs, nc, h_gt, w_gt = gt.size()

    if mask is not None:
        mask = mask.repeat(1, 2, 1, 1)
        gt = torch.where(mask.to(bool), gt, torch.full_like(gt, float('nan')))

    u_gt, v_gt = gt[:,0,:,:], gt[:,1,:,:]
    """
    if scale_to is None:
        pred = torch.nn.functional.upsample(pred, size=(h_gt, w_gt), mode='bilinear')
    else:
        pred = torch.nn.functional.upsample(pred, size=scale_to, mode='bilinear')
        pred = torch.nn.functional.pad(pred, ((w_gt-scale_to[1])//2, (w_gt-scale_to[1]+1)//2, (h_gt-scale_to[0])//2, (h_gt-scale_to[0]+1)//2), mode='constant', value=float('nan'))
    u_pred = pred[:,0,:,:] * (w_gt/w_pred)
    v_pred = pred[:,1,:,:] * (h_gt/h_pred)
    """
    u_pred = pred[:,0,:,:]
    v_pred = pred[:,1,:,:]

    epe = torch.sqrt(torch.pow((u_gt - u_pred), 2) + torch.pow((v_gt - v_pred), 2))

    return epe.nanmean()

def compute_fl(gt, pred, scale_to=None, mask=None, reduce=True, mode='kitti'):
    _, _, h_pred, w_pred = pred.size()
    bs, nc, h_gt, w_gt = gt.size()

    if mask is not None:
        mask = mask.repeat(1, 2, 1, 1)
        gt = torch.where(mask.to(bool), gt, torch.full_like(gt, float('nan')))

    u_gt, v_gt = gt[:,0,:,:], gt[:,1,:,:]
    """
    if scale_to is None:
        pred = torch.nn.functional.upsample(pred, size=(h_gt, w_gt), mode='bilinear')
    else:
        pred = torch.nn.functional.upsample(pred, size=scale_to, mode='bilinear')
        pred = torch.nn.functional.pad(pred, ((w_gt-scale_to[1])//2, (w_gt-scale_to[1]+1)//2, (h_gt-scale_to[0])//2, (h_gt-scale_to[0]+1)//2), mode='constant', value=float('nan'))
    u_pred = pred[:,0,:,:] * (w_gt/w_pred)
    v_pred = pred[:,1,:,:] * (h_gt/h_pred)
    """
    u_pred = pred[:,0,:,:]
    v_pred = pred[:,1,:,:]

    gt_norm = torch.linalg.norm(gt, dim=1, keepdim=True)
    if mode == 'kitti':
        min_bar = torch.maximum(torch.full_like(gt_norm, 3.), gt_norm * 0.05)
    elif mode == 'spring':
        min_bar = torch.full_like(gt_norm, 1.)
    epe = torch.sqrt(torch.pow((u_gt - u_pred), 2) + torch.pow((v_gt - v_pred), 2))[:, None]

    fl = (epe > min_bar).to(float)
    fl = torch.where(torch.isnan(epe), torch.full_like(fl, float('nan')), fl)

    if reduce:
        return fl.nanmean() * 100
    else:
        return torch.nan_to_num(fl)

def upsampled_flow_match(gt, pred):
    gt = gt / 2
    gt = torch.reshape(gt, (gt.shape[0], 2, gt.shape[2]//2, 2, gt.shape[3]//2, 2))
    gt = torch.permute(gt, (0, 1, 2, 4, 3, 5))
    gt = torch.reshape(gt, (*gt.shape[:-2], 4))
    pred = pred[..., None]

    epe = torch.linalg.norm(gt - pred, dim=1, keepdim=True)
    min_idx = torch.argmin(epe, dim=-1, keepdim=True)
    min_idx = min_idx.repeat(1, 2, 1, 1, 1)
    gt = torch.gather(gt, -1, min_idx)

    return gt[..., 0]

def accumulate_metrics(dataset, model_fwd, cfg=None):
    bsz = 1 if cfg is None else cfg.validation.data.batch_size
    dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=bsz,
                num_workers=7,
                shuffle=False,
                collate_fn=Batch.collate_fn
            )

    # start with epe only
    epes = []
    metrics = {'epe': [], 'fl': [], '1px': []} 
    for name in dataset[0].get_dict("masks"):
        metrics['epe_' + name] = []
        metrics['fl_' + name] = []
        metrics['1px_' + name] = []
    if isinstance(dataset, SintelSuperResDataset):
        primary_metric = 'epe'
    elif isinstance(dataset, KITTISuperResDataset):
        primary_metric = 'fl'
    elif isinstance(dataset, SpringSuperResDataset):
        primary_metric = '1px'

    itr = tqdm(dataloader)
    i = 0
    while i < len(dataset):
        try:
            with torch.no_grad():
                itr = tqdm(dataloader)
                j = 0
                for batch in itr:
                    if j < i:
                        j = j + 1
                        continue
                    gt = batch.flow[0]
                    pred = model_fwd(batch)
                    if pred.shape[1] != 2:
                        metrics[primary_metric].append(0.)
                    else:
                        gt = torch.Tensor(gt).to(pred.device)
                        if isinstance(dataset, SpringSuperResDataset):
                            gt = upsampled_flow_match(gt, pred)
            
                        metrics['epe'].append(compute_epe(gt, pred).item())
                        metrics['fl'].append(compute_fl(gt, pred).item())
                        metrics['1px'].append(compute_fl(gt, pred, mode='spring').item())
                        for name, mask in batch.get_dict("masks").items():
                            mask = mask.to(pred.device)
                            metrics['epe_' + name].append(compute_epe(gt, pred, mask=mask).item())
                            metrics['fl_' + name].append(compute_fl(gt, pred, mask=mask).item())
                            metrics['1px_' + name].append(compute_fl(gt, pred, mask=mask, mode='spring').item())
        
                    itr.set_postfix({primary_metric: torch.Tensor(metrics[primary_metric]).mean()})
                    i = i + 1
                    j = j + 1
        except RuntimeError as e:
            print(traceback.format_exc())
    metrics = {name: torch.Tensor(vals) for name, vals in metrics.items()}
    metrics['primary'] = deepcopy(metrics[primary_metric])

    return metrics

def visualize(dataset, model_fwd, sample_idxs=None, display_keys=None, warped_fwd=None):
    # Visualize some samples
    if sample_idxs is None:
        sample_idxs = list(range(16))
    
    if display_keys is None:
        display_keys = ['frame2', 'frame3', 'gt_flow', 'pred_flow', 'diff_flow']
        if len(model_fwd) > 1:
            display_keys += ['pred_flow2', 'diff_flow2', 'diff_between']
        if "occ" in dataset[0].get_dict("masks"):
            display_keys.append('occ')
        if isinstance(dataset, KITTISuperResDataset):
            display_keys.append('inpaint_gt')
            display_keys.append('fl_pos')
        if warped_fwd is not None:
            display_keys += ['warped_f2', 'warped_f2_diff']
    out = {key: [] for key in display_keys}

    for i in sample_idxs:
        with torch.no_grad():
            batch = Batch.collate_fn([dataset[i]])
            pred_flow = []
            for m_fwd in model_fwd:
                pred_flow.append(m_fwd(batch))
        if 'frame2' in display_keys:
            out['frame2'].append(batch.frames_orig[1])
        if 'frame3' in display_keys:
            out['frame3'].append(batch.frames_orig[2])
        if 'gt_flow' in display_keys:
            out['gt_flow'].append(flow_to_image(torch.nan_to_num(batch.flow[0])))
        if 'inpaint_gt' in display_keys:
            points = torch.nonzero(torch.logical_not(torch.isnan(torch.sum(batch.flow[0], dim=1, keepdim=True)))).cpu().numpy()[:, 2:]
            points = np.flip(points, -1)
            values = batch.flow[0][0, 0, points[:, 1], points[:, 0]].cpu().numpy()
            interpolator = LinearNDInterpolator(points, values, fill_value=0.)
            h_inpainted = interpolator(*np.meshgrid(np.arange(0, batch.flow[0].shape[3]), np.arange(0, batch.flow[0].shape[2])))
            values = batch.flow[0][0, 1, points[:, 1], points[:, 0]].cpu().numpy()
            interpolator = LinearNDInterpolator(points, values, fill_value=0.)
            w_inpainted = interpolator(*np.meshgrid(np.arange(0, batch.flow[0].shape[3]), np.arange(0, batch.flow[0].shape[2])))

            inpainted_flow = torch.stack((torch.Tensor(h_inpainted), torch.Tensor(w_inpainted)), dim=0)[None]
            out['inpaint_gt'].append(flow_to_image(inpainted_flow))
        if 'pred_flow' in display_keys:
            if pred_flow[0].shape[1] == 2:
                out['pred_flow'].append(flow_to_image(pred_flow[0]))
            elif pred_flow[0].shape[1] == 1:
                out['pred_flow'].append(pred_flow[0].repeat(1, 3, 1, 1))
            elif pred_flow[0].shape[1] == 3:
                out['pred_flow'].append(pred_flow[0])
                if 'diff_flow' in display_keys:
                    display_keys.remove('diff_flow')
                    display_keys.remove('diff_flow2')
                    display_keys.remove('diff_between')
                out.pop('diff_flow', None)
                out.pop('diff_flow2', None)
                out.pop('diff_between', None)
        if 'pred_flow2' in display_keys:
            out['pred_flow2'].append(flow_to_image(pred_flow[1]))
        if 'diff_flow' in display_keys:
            if isinstance(dataset, SpringSuperResDataset):
                out['diff_flow'].append(flow_to_image(torch.nan_to_num(pred_flow[0] - upsampled_flow_match(batch.flow[0], pred_flow[0]))))
            else:
                out['diff_flow'].append(flow_to_image(torch.nan_to_num(pred_flow[0] - batch.flow[0].to(pred_flow[0].device))))
        if 'diff_flow2' in display_keys:
            if isinstance(dataset, SpringSuperResDataset):
                out['diff_flow2'].append(flow_to_image(torch.nan_to_num(pred_flow[1] - upsampled_flow_match(batch.flow[0], pred_flow[1]))))
            else:
                out['diff_flow2'].append(flow_to_image(torch.nan_to_num(pred_flow[1] - batch.flow[0].to(pred_flow[1].device))))
        if 'diff_between' in display_keys:
            #out['diff_between'].append(flow_to_image(pred_flow[0] - pred_flow[1]))
            error_1 = torch.linalg.norm(pred_flow[0] - batch.flow[0], dim=1, keepdim=True)
            error_2 = torch.linalg.norm(pred_flow[1] - batch.flow[0], dim=1, keepdim=True)
            error_2 = torch.clamp(error_2, min=1e-10)

            #ratio = error_1 / error_2
            NUM_MAX = 10.0
            ratio = error_2 - error_1
            #ratio = torch.clamp(ratio, 1/NUM_MAX, NUM_MAX)
            ratio = torch.clamp(ratio, -NUM_MAX, NUM_MAX)

            green_mask = (ratio > 0.0).float()
            green_vals = ratio/NUM_MAX #(ratio - 1.0)/(NUM_MAX - 1.0)
            red_mask = (ratio < 0.0).float()
            red_vals = -ratio/NUM_MAX #(1/ratio - 1.0)/(NUM_MAX - 1.0)

            ret = torch.ones((error_1.shape[0], 3, error_1.shape[2], error_1.shape[3])).to(error_1.device)
            ret[:, 0] = ret[:, 0] - green_vals * green_mask
            ret[:, 1] = ret[:, 1] - red_vals * red_mask
            ret[:, 2] = ret[:, 2] - green_vals * green_mask - red_vals * red_mask
            out['diff_between'].append(ret)
        if 'occ' in display_keys:
            print(batch.get_dict("masks")) # empty
            out['occ'].append(batch.get_dict("masks")['occ'].repeat(1, 3, 1, 1))
        if 'fl_pos' in display_keys:
            fl_pos = compute_fl(batch.flow[0], pred_flow, 
                                scale_to=batch.frames_up[0].shape[2:], 
                                reduce=False)
            out['fl_pos'].append(fl_pos.repeat(1, 3, 1, 1))

    max_size = (max([log.shape[2] for key, val in out.items() for log in val]), 
                max([log.shape[3] for key, val in out.items() for log in val]))
    for key in out:
        for i in range(len(out[key])):
            out[key][i] = torchvision.transforms.functional.resize(out[key][i], max_size).detach().cpu()
        out[key] = torch.concatenate(out[key], dim=0)
    out = torch.stack([out[key] for key in display_keys], dim=1)

    out = torch.reshape(out, (len(sample_idxs) * len(display_keys), 3, *max_size))
    return torchvision.utils.make_grid(out, 
                                       nrow=len(display_keys),
                                       normalize=True,
                                       scale_each=True)

def evaluate(dataset, model_fwd, cfg=None, warped_fwd=None): # eventually support model fwd
    metrics = accumulate_metrics(dataset, model_fwd, cfg=cfg)
    random_idx = list(range(0, len(dataset), len(dataset)//16))
    image_random = visualize(dataset, [model_fwd], sample_idxs=random_idx)

    worst = torch.topk(metrics['primary'], 8, sorted=True).indices
    best = torch.topk(metrics['primary'], 8, largest=False, sorted=True).indices
    image_worst = visualize(dataset, [model_fwd], sample_idxs=worst)
    image_best = visualize(dataset, [model_fwd], sample_idxs=best)

    return {
        'metrics': metrics,
        'random': image_random,
        'worst': image_worst,
        'best': image_best,
    }

def evaluate_against(dataset, model_fwd_1, model_fwd_2, cfg=None): # eventually support model fwd
    metrics_1 = accumulate_metrics(dataset, model_fwd_1, cfg=cfg)
    metrics_2 = accumulate_metrics(dataset, model_fwd_2, cfg=cfg)
    if len(dataset) < 16:
        random_idx = list(range(0, len(dataset)))
    else:
        random_idx = list(range(0, len(dataset), len(dataset)//16))
    image_random = visualize(dataset, [model_fwd_1, model_fwd_2], sample_idxs=random_idx)

    num_comp = min(8, len(dataset))
    worst = torch.topk(metrics_1['primary'], num_comp, sorted=True).indices
    best = torch.topk(metrics_1['primary'], num_comp, largest=False, sorted=True).indices
    worst_comp = torch.topk(metrics_1['primary'] - metrics_2['primary'], num_comp, sorted=True).indices
    best_comp = torch.topk(metrics_1['primary'] - metrics_2['primary'], num_comp, largest=False, sorted=True).indices
    image_worst = visualize(dataset, [model_fwd_1, model_fwd_2], sample_idxs=worst)
    image_best = visualize(dataset, [model_fwd_1, model_fwd_2], sample_idxs=best)
    image_worst_comp = visualize(dataset, [model_fwd_1, model_fwd_2], sample_idxs=worst)
    image_best_comp = visualize(dataset, [model_fwd_1, model_fwd_2], sample_idxs=best)

    return {
        'metrics': metrics_1,
        'metrics2': metrics_2,
        'random': image_random,
        'worst': image_worst,
        'best': image_best,
        'worst_comp': image_worst_comp,
        'best_comp': image_best_comp,
    }

def raft_round(img, ret_pad=False, is_sintel=False):
    ht, wd = img.shape[-2:]
    pad_ht = (((ht // 8) + 1) * 8 - ht) % 8
    pad_wd = (((wd // 8) + 1) * 8 - wd) % 8
    if is_sintel:
        pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
    else:
        pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]
    if ret_pad:
        return pad

    return torch.nn.functional.pad(img, pad, mode='replicate')

def raft_unpad(img, pad): 
    ht, wd = img.shape[-2:]
    c = [pad[2], ht-pad[3], pad[0], wd-pad[1]]
    return img[..., c[0]:c[1], c[2]:c[3]]

def get_raft_fwd(dataset, ckpt=None, device=None, cfg=None):
    class RAFTargs():
        def __init__(self):
            self.alternate_corr = False
            self.small = False
            self.mixed_precision = False
            if ckpt is None:
                if isinstance(dataset, SintelSuperResDataset) or isinstance(dataset, SpringSuperResDataset):
                    self.dataset = "sintel"
                elif isinstance(dataset, KITTISuperResDataset):
                    self.dataset = "kitti"
                self.model = f"RAFT/models/raft-{self.dataset}.pth"
            else:
                self.model = f"RAFT/models/raft-{ckpt}.pth"

        def __iter__(self):
            for k, v in vars(self).items():
                yield k
    args = RAFTargs()
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model.to('cuda')
    model.eval()

    def raft_fwd(b):
        b.to(device)
        pad = raft_round(b.frames_orig[1], ret_pad=True)
        raft_out = model(255 * raft_round(b.frames_orig[1], is_sintel=isinstance(dataset, SintelSuperResDataset)),
                         255 * raft_round(b.frames_orig[2], is_sintel=isinstance(dataset, SintelSuperResDataset)), 
                         iters=24 if isinstance(dataset, KITTISuperResDataset) else 32, 
                         test_mode=True)[1]
        """
        raft_out = model(raft_round(b.frames_orig[1], is_sintel=isinstance(dataset, SintelSuperResDataset)),
                         raft_round(b.frames_orig[2], is_sintel=isinstance(dataset, SintelSuperResDataset)), 
                         num_flow_updates=32)[-1]
        """
        ret = raft_unpad(raft_out, pad)
        return ret
    #return evaluate(dataset, raft_fwd, cfg=cfg)
    return raft_fwd

def get_smurf_fwd(dataset, checkpoint=None, cfg=None, device=None):
    if checkpoint is None:
        if isinstance(dataset, SintelSuperResDataset) or isinstance(dataset, SpringSuperResDataset):
            checkpoint = "smurf_sintel.pt"
        elif isinstance(dataset, KITTISuperResDataset):
            checkpoint = "smurf_kitti.pt"
    print(f'Using SMURF checkpoint {checkpoint}...')

    smurf = raft_smurf(checkpoint=checkpoint)
    if device is not None:
        smurf.to(device)
    smurf.eval()

    if isinstance(dataset, SintelSuperResDataset) or isinstance(dataset, SpringSuperResDataset):
        inference_shape = (480, 928)
    elif isinstance(dataset, KITTISuperResDataset):
        inference_shape = (488, 1144)
    def smurf_fwd(b):
        b.to(device)

        to_inference_shape = torchvision.transforms.Resize(inference_shape, antialias=True)
        from_inference_shape = torchvision.transforms.Resize(b.frames_orig[1].shape[2:], antialias=True)
        flow_multiplier = [b.frames_orig[1].shape[3] / inference_shape[1], b.frames_orig[1].shape[2] / inference_shape[0]]
        flow_multiplier = torch.Tensor(flow_multiplier)[None,  :, None, None].to(device)

        frame1, frame2 = to_inference_shape(b.frames_orig[1]), to_inference_shape(b.frames_orig[2])
        smurf_out = smurf(frame1, frame2, num_flow_updates=12)[-1]
        smurf_out = from_inference_shape(smurf_out) * flow_multiplier

        return smurf_out
    #return evaluate(dataset, smurf_fwd, cfg=cfg)
    return smurf_fwd

class InstanceDataset():
    def __init__(self, batch):
        self.batch = batch

    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return self.batch

def get_overfit_fwd(dataset, device=None, cfg=None):
    num_fit = 8000
    data_dir = "logs/overfit_fwd/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if len(os.listdir(data_dir)) > 0:
        #raise ValueError("data dir not empty for overfit")
        input("The overfit save path already exists, continue?")

    # Set image size
    _ = dataset[0]
    cfg.dataset.imsz = str(dataset.imsz[0]) + "," + str(dataset.imsz[1])
    cfg.dataset.imsz_super = str(dataset.imsz_super[0]) + "," + str(dataset.imsz_super[1])

    def overfit_fwd(b):
        if b.frames[0].shape[0] != 1:
           raise ValueError("overfit only works on batches of size 1")
        h = hashlib.new('md5')
        h.update(b.path[0].encode())
        hash_path = h.hexdigest()

        if not os.path.exists(data_dir + hash_path + "/flow.pt"):
            os.makedirs(data_dir + hash_path)
            with open(data_dir + hash_path + "/out.txt", "w") as ff:#sys.stdout:
                with open(data_dir + hash_path + "/err.txt", "w") as gg:#sys.stderr:
                    model = OverfitSoftLearner(cfg)
                    instance_dataset = InstanceDataset(b)
                    dataloader = torch.utils.data.DataLoader(
                        instance_dataset,
                        batch_size=1,
                        shuffle=False,
                        collate_fn=lambda x: x[0],
                    )

                    trainer = L.Trainer(
                        max_epochs=num_fit,
                        accelerator='auto',
                        logger = False, 
                        devices="auto",
                        precision=cfg.training.precision,
                        check_val_every_n_epoch=cfg.validation.check_epoch,
                        overfit_batches=1,
                        limit_val_batches=cfg.validation.limit_batch,
                    )
                    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
                    logging.getLogger("lightning").setLevel(logging.ERROR)
                    """WandbLogger(
                        project=cfg.wandb.project,
                        mode="offline",
                        name=cfg.wandb.name+"_"+b.path[0]
                    ),
                    """

                    trainer.fit(
                        model,
                        train_dataloaders=dataloader,
                        val_dataloaders=dataloader,
                    )

                    #wandb.finish()
            # save the model and predicted flow
            final_flow = model.model(b, temp=model.temp, src_idx=[2], tgt_idx=[1])["positions"][:, 0].permute(0, 3, 1, 2).float()
            torch.save(model.state_dict(), data_dir + hash_path + "/model.pt")
            torch.save(final_flow, data_dir + hash_path + "/flow.pt")
            torch.save(b, data_dir + hash_path + "/batch.pt")
        else:
            final_flow = torch.load(data_dir + hash_path + "/flow.pt")
        
        assert(final_flow.shape[2] / b.frames_up[1].shape[2] == final_flow.shape[3] / b.frames_up[1].shape[3])
        final_flow = torchvision.transforms.functional.resize(final_flow, b.frames_up[1].shape[2:]) * (b.frames_up[1].shape[3] / final_flow.shape[3])
        final_flow = torch.nn.functional.pad(final_flow, dataset.image_trim, mode='constant', value=float('nan'))
        return final_flow

    return overfit_fwd

def run_raft_eval():
    cfg = OmegaConf.load('eval.yaml')
    dataset = SpringSuperResDataset(cfg, "all")
    return evaluate(dataset, get_raft_fwd(
        dataset,
        cfg=cfg,
        device='cuda'
    ), cfg=cfg)

def run_smurf_eval():
    cfg = OmegaConf.load('eval.yaml')
    dataset = SpringSuperResDataset(cfg, "all")
    return evaluate(dataset, get_smurf_fwd(
        dataset,
        cfg=cfg,
        device='cuda'
    ), cfg=cfg)

def run_overfit_eval():
    cfg = OmegaConf.load('eval_overfit.yaml')
    dataset = SintelSuperResDataset(cfg, "validation")
    return evaluate(dataset, get_overfit_fwd(
        dataset,
        cfg=cfg,
        device='cuda'
    ), cfg=cfg)

def run_smurf_against_raft():
    cfg = OmegaConf.load('eval.yaml')
    dataset = SintelSuperResDataset(cfg, "all")
    return evaluate_against(dataset, get_smurf_fwd(
        dataset,
        cfg=cfg,
        device='cuda'
    ), get_raft_fwd(
        dataset,
        cfg=cfg,
        device='cuda'
    ), cfg=cfg)

def run_overfit_against_smurf():
    cfg = OmegaConf.load('eval.yaml')
    overfit_cfg = OmegaConf.load('eval_overfit.yaml')
    dataset = SintelSuperResDataset(cfg, "all")
    return evaluate_against(dataset, get_overfit_fwd(
        dataset,
        cfg=overfit_cfg,
        device='cuda'
    ), get_smurf_fwd(
        dataset,
        cfg=cfg,
        device='cuda'
    ), cfg=cfg)
