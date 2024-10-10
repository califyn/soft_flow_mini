import torch
import torchvision
import torch.nn.functional as F

from copy import deepcopy
import numpy as np

def warp_previous_flow(flow_fields):
    batch_size, n_frames, height, width, _ = flow_fields.shape

    flow_t_plus_1 = flow_fields[:, 1:].reshape(-1, height, width, 2)

    grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width))
    grid = torch.stack((grid_x, grid_y), dim=-1).float().to(flow_t_plus_1.device)
    grid = grid.unsqueeze(0).repeat(batch_size * (n_frames - 1), 1, 1, 1)

    new_positions = grid + flow_t_plus_1
    new_positions = torch.clamp(new_positions, -1, 1)

    flow_fields_t = flow_fields[:, :-1].permute(0, 1, 4, 2, 3).reshape(-1, 2, height, width)

    warped_flow_t = torch.nn.functional.grid_sample(flow_fields_t, new_positions)

    # Reshape it back to original dimensions
    warped_flow_t = warped_flow_t.permute(0, 2, 3, 1).reshape(batch_size, n_frames - 1, height, width, 2)

    return warped_flow_t

def flow_from_weights(weights):
    print('note this is wrong for downsampling flows')
    weight_sl = weights.shape[-1]
    t = weight_sl//2
    
    x_positions = torch.linspace(-t, t, weight_sl)
    y_positions = torch.linspace(-t, t, weight_sl)
    grid_x, grid_y = torch.meshgrid(x_positions, y_positions)
    grid = torch.stack((grid_x, grid_y), dim=-1).to(weights.device)
    #grid *= grid * (-1)

    expected_positions = torch.einsum('bijkl,klm->bijm', weights, grid).flip(-1)
    expected_positions = expected_positions.permute(0, 3, 1, 2)
    return expected_positions

def filter_to_image(filters, downsample_factor=1):
    filters = filters[:, ::downsample_factor, ::downsample_factor]

    # b, h, w, x, y
    height, width, radius = filters.shape[1], filters.shape[2], filters.shape[3]
    filters = filters.reshape((filters.shape[0], -1, radius, radius)) # b (hw) x y

    filters = filters[:, :, None].tile(1, 3, 1, 1)  # b (hw) c x y
    filters[:, :, 2, radius // 2, :] = 0.33 * (1 + 2 * filters[:, :, 2, radius // 2, :])  # tint centers
    filters[:, :, 1, :, radius // 2] = 0.33 * (1 + 2 * filters[:, :, 1, :, radius // 2])
    filters = filters.repeat_interleave(3, dim=3).repeat_interleave(3, dim=4)  # scale up

    return torch.stack([torchvision.utils.make_grid(fil, nrow=width, pad_value=1.0) for fil in filters], dim=0)

def flow_to_filter(flow, weight_sl):
    print('flow to filter not for downsampling flows')
    t = weight_sl//2
    positions = torch.linspace(-t, t, weight_sl).to(flow.device)
    #flow = flow.flip(1)

    x_diff = flow[:, 0, :, :, None] - positions[None, None, None, :]
    y_diff = flow[:, 1, :, :, None] - positions[None, None, None, :]
    x_diff = 1 - torch.minimum(torch.abs(x_diff), torch.ones_like(x_diff))
    y_diff = 1 - torch.minimum(torch.abs(y_diff), torch.ones_like(y_diff))

    filters = x_diff[:, :, :, None, :] * y_diff[:, :, :, :, None]

    return filters

"""
def quiver_plot(flow):
    height, width = model_output['positions'].shape[-3:-1]
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    Y = height - Y
    step = 5  # Reduce this value if you want more arrows

    fig, axs = plt.subplots(6, n_frames, figsize=(5 * n_frames, 6 * 5))  # 4 rows, n_frames columns

    for i in range(n_frames - 1):
        expected_pos = model_output['positions'][batch_index, i].cpu().detach().numpy()
        U = expected_pos[:, :, 1] * width
        V = expected_pos[:, :, 0] * height
        # mask = np.logical_or(U > 1, V > 1)
        axs[5, i+1].quiver(X[::step, ::step], Y[::step, ::step], U[::step, ::step], V[::step, ::step], angles='xy', scale_units='xy', scale=1, color='r')
        # axs[5, i+1].imshow(in_frames[i + 1])  # Show the frame where the correspondence is indicated
        axs[5, i+1].axis('off')
        axs[5, i+1].set_title(f"Quiver Plot {i+2}")
        axs[5, i+1].set_aspect('equal')
"""

def pad_for_filter(in_frames, weight_sl, downsample_factor, border_fourth_channel=False, pad=True, unfold=True):
    t = weight_sl//2 # smaller t for padding
    in_shape = in_frames.shape

    if pad:
        if border_fourth_channel:
            in_shape_fourth = list(in_frames.shape)
            in_shape_fourth[2] = 1

            in_frames_fourth = in_frames.new_ones(tuple(in_shape_fourth))
            #in_frames = torch.cat((in_frames, in_frames_fourth), dim=2)

        in_frames_reshaped = torch.reshape(in_frames, (in_frames.shape[0] * in_frames.shape[1], in_frames.shape[2], in_frames.shape[3], in_frames.shape[4]))
        x_padded = F.pad(in_frames_reshaped, (t, t, t, t), mode='replicate')
        if border_fourth_channel:
            in_frames_fourth_reshaped = torch.reshape(in_frames_fourth, (in_frames_fourth.shape[0] * in_frames_fourth.shape[1], in_frames_fourth.shape[2], in_frames_fourth.shape[3], in_frames_fourth.shape[4]))
            x_padded_fourth = F.pad(in_frames_fourth_reshaped, (t, t, t, t), mode='constant', value=0)
            x_padded = torch.cat((x_padded, x_padded_fourth), dim=1)
        #if border_fourth_channel:
        #    x_padded[:, -1, :, :] = torch.ones_like(x_padded[:, -1, :, :])
        x_padded = torch.reshape(x_padded, (in_shape[0], in_shape[1], *x_padded.shape[1:]))
    else:
        x_padded = in_frames

    if unfold:
        x_padded = x_padded.unfold(3, weight_sl + downsample_factor - 1, downsample_factor).unfold(4, weight_sl + downsample_factor - 1, downsample_factor)  
        #x_padded_alt = torch.nn.Unfold(

    return x_padded

def bound_mask(flow):
    R = flow.shape[-3]
    H, W = flow.shape[-2], flow.shape[-1]

    mask = torch.nn.functional.unfold(
            torch.ones((1, H, W)),
            (R, R),
            padding=R // 2
    ).reshape((1, 1, R, R, H, W)).to(flow.device)

    return mask

def downsample_filter(fil, downsample_factor=1):
    fil = torch.reshape(fil, (fil.shape[0], 
                         fil.shape[1], 
                         fil.shape[2], 
                         fil.shape[3], 
                         fil.shape[4]//downsample_factor, downsample_factor, 
                         fil.shape[5]//downsample_factor, downsample_factor))
    fil = torch.sum(fil, dim=(5, 7))
    return fil

def transpose_filter(fil, downsample_factor=1):
    flow = fil.clone()
    flow = downsample_filter(flow, downsample_factor=downsample_factor)
    flow = torch.permute(flow, (0, 1, 4, 5, 2, 3))

    R = flow.shape[-3]
    H, W = flow.shape[-2], flow.shape[-1]
    fil_size = (1, 1, R, R, H, W)
    mask = bound_mask(flow).to('cpu')
    
    dx = (torch.arange(0, R) - R // 2)[None, None, :, None, None, None]
    x_ = torch.arange(0, H)[None, None, None, None, :, None]
    dx = torch.broadcast_to(dx, fil_size).clone()
    x_ = torch.broadcast_to(x_, fil_size).clone()
    x = dx + x_

    dy = (torch.arange(0, R) - R // 2)[None, None, None, :, None, None]
    y_ = torch.arange(0, W)[None, None, None, None, None, :]
    dy = torch.broadcast_to(dy, fil_size).clone()
    y_ = torch.broadcast_to(y_, fil_size).clone()
    y = dy + y_

    idxs = torch.stack((dx, x_, x, dy, y_, y, mask), dim=0)
    idxs = idxs.reshape((7, -1)).long().to(flow.device)
    idxs = idxs[:, idxs[6] == 1]
    dx, x_, x, dy, y_, y, _ = tuple(torch.chunk(idxs, 7, dim=0))

    t_flow = torch.zeros_like(flow)
    t_flow[:, :, R // 2 - dx, R // 2 - dy, x, y] = flow[:, :, R // 2 + dx, R // 2 + dy, x_, y_]
    t_flow = torch.permute(t_flow, (0, 1, 4, 5, 2, 3))
    return t_flow

def tiled_pred(model, batch, flow_max, crop_batch_fn, crop=(224, 224), temp=9, out_key='flow', loss_fn=None, given_dim=1, do_att_map=True):
    # this could be a lot better in a lot of ways, but it should work for now
    H, W = batch.frames[0].shape[2], batch.frames[0].shape[3]
    assert(crop[0] <= H and crop[1] <= W)

    x_step = crop[0] - 2 * flow_max
    x_idxs = [0, crop[0] - flow_max]
    middle_steps = list(range(crop[0] - flow_max + x_step, H - (crop[0] - flow_max), x_step))
    x_idxs += middle_steps + [H - (crop[0] - flow_max), H]

    y_step = crop[1] - 2 * flow_max
    y_idxs = [0, crop[1] - flow_max]
    middle_steps = list(range(crop[1] - flow_max + y_step, W - (crop[1] - flow_max), y_step))
    y_idxs += middle_steps + [W - (crop[1] - flow_max), W]
   
    def generate_crop_from_region(x1, x2, y1, y2):
        x_leftover = crop[0] - (x2 - x1)
        y_leftover = crop[1] - (y2 - y1)
        
        x1_margin = x1
        x2_margin = H - x2
        y1_margin = y1
        y2_margin = W - y2

        border = [x_leftover//2, (x_leftover + 1)//2, y_leftover//2, (y_leftover+1)//2]
        if border[0] > x1_margin:
            border[0] = x1_margin
            border[1] = x_leftover - border[0]
        elif border[1] > x2_margin:
            border[1] = x2_margin
            border[0] = x_leftover - border[1] 
        if border[2] > y1_margin:
            border[2] = y1_margin
            border[3] = y_leftover - border[2]
        elif border[3] > y2_margin:
            border[3] = y2_margin
            border[2] = y_leftover - border[3] 
        
        patch_crop = [x1 - border[0], H - (x2 + border[1]), y1 - border[2], W - (y2 + border[3])]
        def get_estimate_fn(img):
            img = img[..., border[0]:, border[2]:]
            if border[1] > 0:
                img = img[..., :-border[1], :]
            if border[3] > 0:
                img = img[..., :, :-border[3]]
            return img
        return tuple(patch_crop), get_estimate_fn

    if out_key == 'flow':
        estimate = torch.full_like(batch.flow[0], float('nan'))
    elif out_key == 'pred_occ_mask':
        estimate = torch.full_like(batch.flow[0][:, 0, None], float('nan'))
    elif out_key == 'given':
        estimate = torch.full_like(batch.flow[0][:, 0, None], float('nan')).repeat(1, given_dim, 1, 1)
    for x_region in zip(x_idxs[:-1], x_idxs[1:]):
        for y_region in zip(y_idxs[:-1], y_idxs[1:]):
            patch_crop, get_estimate_fn = generate_crop_from_region(*x_region, *y_region)
            cropped_batch = crop_batch_fn(batch, patch_crop)
            
            if out_key == 'flow':
                out_flow = model(cropped_batch, temp=temp, src_idx=[2], tgt_idx=[1])
                out_flow = out_flow["positions"][:, 0].permute(0, 3, 1, 2).float()
            elif out_key == 'pred_occ_mask':
                out = model(cropped_batch, temp=temp, src_idx=[1], tgt_idx=[2])
                if out["pred_occ_mask"] is not None:
                    out_flow = 1. - out["pred_occ_mask"][:, 0].float()
                else:
                    out_flow = torch.ones((out["positions"].shape[0], 1, out["positions"].shape[2], out["positions"].shape[3])).to(out["positions"].device)
                if model.border_handling in ["pad_feats", "fourth_channel"]:
                    out_border = model(cropped_batch, temp=temp, src_idx=[2], tgt_idx=[1])
                    border_weights = out_border["out"][:, 0, -1, None].detach()
                    #border_weights = 1. - torch.gt(border_weights, torch.Tensor([0.5]).to(border_weights.device)).float()
                    border_weights = torch.gt(border_weights, torch.Tensor([0.5]).to(border_weights.device)).float()
                    #print(out_flow.shape, border_weights.shape, out_flow.dtype, border_weights.dtype, out_flow[..., :5, :5], border_weights[..., :5, :5], (out_flow * border_weights)[..., :5, :5])
                    out_flow = out_flow * border_weights # border weights mean is 0.027
            elif out_key == 'given':
               out_flow = model(cropped_batch)
            out_flow = get_estimate_fn(out_flow)
            
            estimate[..., x_region[0]:x_region[1], y_region[0]:y_region[1]] = out_flow

    if estimate.shape[1] > 5 and do_att_map:
        query_idx = estimate.shape[2] // 2, int(round(estimate.shape[3] * 0.35)) #estimate.shape[3] // 2#
        feat_dim = estimate.shape[1] // 2
        query = estimate[:, feat_dim:, query_idx[0], query_idx[1]]

        estimate = torch.sum(estimate[:, :feat_dim] * query[:, :, None, None], dim=1, keepdim=True)

        estimate_shape = estimate.shape
        estimate = estimate.reshape((estimate.shape[0], estimate.shape[1], -1))
        #estimate = torch.nn.functional.softmax(estimate * temp / 2, dim=-1)
        estimate = estimate.reshape(estimate_shape)

        estimate = estimate.repeat(1, 2, 1, 1)

        """
        diff_query = batch.flow[0][0, :, query_idx[0], query_idx[1]]
        query_idx = int(round(query_idx[0] + diff_query[1].item())), int(round(query_idx[1] + diff_query[0].item()))
        estimate[:, 1, query_idx[0]-2, :] = 1 - (1 - estimate[:, 1, query_idx[0]-2, :]) * 0.8
        estimate[:, 1, query_idx[0]+2, :] = 1 - (1 - estimate[:, 1, query_idx[0]+2, :]) * 0.8
        estimate[:, 2, :, query_idx[1]-2] = 1 - (1 - estimate[:, 2, :, query_idx[1]-2]) * 0.8
        estimate[:, 2, :, query_idx[1]+2] = 1 - (1 - estimate[:, 2, :, query_idx[1]+2]) * 0.8
        """

    return estimate

# tiled prediction, taken from Croco v2
# https://github.com/naver/croco/blob/master/stereoflow/engine.py#L179 
# edit: not using this for now, it's a bit too unwieldy/unpredictable
@torch.no_grad()
def other_tiled_pred(model, criterion, batch,
               overlap=0.5, crop=(224, 224), temp=9,
               conf_mode='conf_expsigmoid_10_5', with_conf=False, 
               return_time=False):
    if batch.uncropped_batch is not None:
        batch = batch.uncropped_batch
    
    # for each image, we are going to run inference on many overlapping patches
    # then, all predictions will be weighted-averaged
    if batch.flow is not None:
        B, C, H, W = batch.flow[0].shape
    else:
        B, _, H, W = batch.frames[1].shape
        C = model.head.num_channels-int(with_conf)
    win_height, win_width = crop[0], crop[1]
    
    # upscale to be larger than the crop
    do_change_scale =  H<win_height or W<win_width
    if do_change_scale: 
        raise ValueError("Image cannot be smaller than the crop")
        """
        upscale_factor = max(win_width/W, win_height/W)
        original_size = (H,W)
        new_size = (round(H*upscale_factor),round(W*upscale_factor))
        img1 = _resize_img(img1, new_size)
        img2 = _resize_img(img2, new_size)
        # resize gt just for the computation of tiled losses
        if batch.flow[0] is not None: batch.flow[0] = _resize_stereo_or_flow(batch.f, new_size)
        H,W = img1.shape[2:4]
        """
        
    if conf_mode.startswith('conf_expsigmoid_'): # conf_expsigmoid_30_10
        beta, betasigmoid = map(float, conf_mode[len('conf_expsigmoid_'):].split('_'))
    elif conf_mode.startswith('conf_expbeta'): # conf_expbeta3
        beta = float(conf_mode[len('conf_expbeta'):])
    else:
        raise NotImplementedError(f"conf_mode {conf_mode} is not implemented")

    def crop_generator():
        for sy in _overlapping(H, win_height, overlap):
          for sx in _overlapping(W, win_width, overlap):
            yield sy, sx, sy, sx, True

    # keep track of weighted sum of prediction*weights and weights
    accu_pred = batch.frames[1].new_zeros((B, C, H, W)) # accumulate the weighted sum of predictions 
    accu_conf = batch.frames[1].new_zeros((B, H, W)) + 1e-16 # accumulate the weights 
    accu_c = batch.frames[1].new_zeros((B, H, W)) # accumulate the weighted sum of confidences ; not so useful except for computing some losses

    tiled_losses = []
    
    if return_time:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    for sy1, sx1, sy2, sx2, aligned in crop_generator():
        # compute optical flow there
        cropped_batch = deepcopy(batch)
        """
        cropped_batch.frames[0] = _crop(batch.frames[0],sy1,sx1) # we don't need this, but just for model internals to work
        cropped_batch.frames[1] = _crop(batch.frames[1],sy1,sx1)
        cropped_batch.frames[2] = _crop(batch.frames[2],sy2,sx2)
        """
        pred =  model(cropped_batch, temp=temp, src_idx=[2], tgt_idx=[1])
        pred = pred["positions"][:, 0].permute(0, 3, 1, 2).float()
        #pred, predconf = split_prediction_conf(pred, with_conf=with_conf)
        predconf = None
        
        if batch.flow is not None: gtcrop = _crop(batch.flow[0],sy1,sx1)
        if criterion is not None and batch.flow is not None: 
            tiled_losses.append( criterion(pred, gtcrop).item() if predconf is None else criterion(pred, gtcrop, predconf).item() )
        
        if predconf is not None:
            if conf_mode.startswith('conf_expsigmoid_'):
                conf = torch.exp(- beta * 2 * (torch.sigmoid(predconf / betasigmoid) - 0.5)).view(B,win_height,win_width)
            elif conf_mode.startswith('conf_expbeta'):
                conf = torch.exp(- beta * predconf).view(B,win_height,win_width)
            else:
                raise NotImplementedError
        else:
            conf = torch.ones((B, win_height, win_width)).to(pred.device)
                        
        accu_pred[...,sy1,sx1] += pred * conf[:,None,:,:]
        accu_conf[...,sy1,sx1] += conf
        if predconf is not None:
            accu_c[...,sy1,sx1] += predconf.view(B,win_height,win_width) * conf 
        else:
            accu_c[...,sy1,sx1] += conf 
        
    pred = accu_pred / accu_conf[:, None,:,:]
    c = accu_c / accu_conf
    assert not torch.any(torch.isnan(pred))

    if return_time:
        end.record()
        torch.cuda.synchronize()
        time = start.elapsed_time(end)/1000.0 # this was in milliseconds

    if do_change_scale:
        pred = _resize_stereo_or_flow(pred, original_size)
    
    if return_time:
        return pred, torch.mean(torch.tensor(tiled_losses)), c, time
    return pred, torch.mean(torch.tensor(tiled_losses)), c


def _overlapping(total, window, overlap=0.5):
    assert total >= window and 0 <= overlap < 1, (total, window, overlap)
    num_windows = 1 + int(np.ceil( (total - window) / ((1-overlap) * window) ))
    offsets = np.linspace(0, total-window, num_windows).round().astype(int)
    yield from (slice(x, x+window) for x in offsets)

def _crop(img, sy, sx):
    B, THREE, H, W = img.shape
    if 0 <= sy.start and sy.stop <= H and 0 <= sx.start and sx.stop <= W:
        return img[:,:,sy,sx]
    l, r = max(0,-sx.start), max(0,sx.stop-W)
    t, b = max(0,-sy.start), max(0,sy.stop-H)
    img = torch.nn.functional.pad(img, (l,r,t,b), mode='constant')
    return img[:, :, slice(sy.start+t,sy.stop+t), slice(sx.start+l,sx.stop+l)]
