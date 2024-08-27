import torch
import torchvision
import torch.nn.functional as F

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

def pad_for_filter(in_frames, weight_sl, downsample_factor):
    t = weight_sl//2 # smaller t for padding
    in_shape = in_frames.shape
    in_frames_reshaped = torch.reshape(in_frames, (in_frames.shape[0] * in_frames.shape[1], in_frames.shape[2], in_frames.shape[3], in_frames.shape[4]))
    x_padded = F.pad(in_frames_reshaped, (t, t, t, t), mode='replicate', value=0)
    x_padded = torch.reshape(x_padded, (in_shape[0], in_shape[1], *x_padded.shape[1:]))
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

    flow[:, :, R // 2 - dx, R // 2 - dy, x, y] = flow[:, :, R // 2 + dx, R // 2 + dy, x_, y_]
    flow = torch.permute(flow, (0, 1, 4, 5, 2, 3))
    return flow
