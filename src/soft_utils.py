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

        in_frames_reshaped = torch.reshape(in_frames, (in_frames.shape[0] * in_frames.shape[1], *in_frames.shape[2:]))
        x_padded = F.pad(in_frames_reshaped, (t, t, t, t), mode='replicate')
        if border_fourth_channel:
            in_frames_fourth_reshaped = torch.reshape(in_frames_fourth, (in_frames_fourth.shape[0] * in_frames_fourth.shape[1], *in_frames_fourth.shape[2:]))
            x_padded_fourth = F.pad(in_frames_fourth_reshaped, (t, t, t, t), mode='constant', value=0)
            x_padded = torch.cat((x_padded, x_padded_fourth), dim=1)
        x_padded = torch.reshape(x_padded, (in_shape[0], in_shape[1], *x_padded.shape[1:]))
    else:
        x_padded = in_frames

    if unfold:
        x_padded = x_padded.unfold(3, weight_sl + downsample_factor - 1, downsample_factor).unfold(4, weight_sl + downsample_factor - 1, downsample_factor)  

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

"""
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
"""

def get_range(fil):
    weight_sl = fil.shape[-1]
    t = weight_sl // 2

    fold = torch.nn.Fold((fil.shape[2] + t*2, fil.shape[3] + t*2), weight_sl)
    fil = fil.permute(0, 1, 4, 5, 2, 3).reshape(fil.shape[0], fil.shape[1]*fil.shape[4]*fil.shape[5], fil.shape[2]*fil.shape[3])

    fil_range = fold(fil)
    fil_range = fil_range[..., t:-t, t:-t][:, :, None]
    return fil_range

def tiled_pred(model, batch, flow_max, crop_batch_fn, crop=(224, 224), temp=9, out_key='flow', loss_fn=None, given_dim=1, do_att_map=True, model_fn=None):
    if model_fn is None:
        model_fn = model
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
    for i, x_region in enumerate(zip(x_idxs[:-1], x_idxs[1:])):
        for j, y_region in enumerate(zip(y_idxs[:-1], y_idxs[1:])):
            patch_crop, get_estimate_fn = generate_crop_from_region(*x_region, *y_region)
            cropped_batch = crop_batch_fn(batch, patch_crop)
            
            if out_key == 'flow':
                if i == 0 or i == len(x_idxs) - 2 or j == 0 or j == len(y_idxs) - 2: # border cases
                    out_flow = model_fn(cropped_batch, temp=temp, src_idx=[2], tgt_idx=[1], so_temp=temp)
                else:
                    out_flow = model_fn(cropped_batch, temp=temp, src_idx=[2], tgt_idx=[1], so_temp=temp, no_pad=True)
                out_flow = out_flow["flow"][:, 0]
            elif out_key == 'pred_occ_mask':
                raise ValueError
                out = model_fn(cropped_batch, temp=temp, src_idx=[1], tgt_idx=[2])
                if out["pred_occ_mask"] is not None:
                    out_flow = 1. - out["pred_occ_mask"][:, 0].float()
                else:
                    out_flow = torch.ones((out["flow"].shape[0], 1, out["flow"].shape[2], out["flow"].shape[3])).to(out["flow"].device)
                if model.border_handling in ["pad_feats", "fourth_channel"]:
                    out_border = model_fn(cropped_batch, temp=temp, src_idx=[2], tgt_idx=[1])
                    border_weights = out_border["out"][:, 0, -1, None].detach()
                    border_weights = torch.gt(border_weights, torch.Tensor([0.5]).to(border_weights.device)).float()
                    out_flow = out_flow * border_weights
            elif out_key == 'given':
               out_flow = model_fn(cropped_batch)
            out_flow = get_estimate_fn(out_flow)
            
            estimate[..., x_region[0]:x_region[1], y_region[0]:y_region[1]] = out_flow
    assert(not torch.any(torch.isnan(estimate)))

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
