import torch.distributions as dist
import torch
from .soft_utils import warp_previous_flow, pad_for_filter, transpose_filter, downsample_filter
from tqdm import tqdm

def entropy_loss(weight_softmax):
    entropy = -torch.mean(weight_softmax * torch.log(weight_softmax + 1e-8))
    return entropy

def temporal_smoothness_loss(flow_fields, r=None):
    #expected_flow_t = warp_previous_flow(flow_fields)
    # expected_flow_t = warp_previous_flow_multi(flow_fields)

    #flow_t_plus_1 = flow_fields[:, 1:].reshape(-1, *flow_fields.shape[2:])
    #flow_diffs = flow_t_plus_1 - expected_flow_t
    #loss = (flow_diffs ** 2).mean()
    loss = torch.abs(flow_fields[:, 0] + flow_fields[:, 1])
    if r is not None:
        weights = torch.abs(flow_fields[:, 0] - flow_fields[:, 1]) * 0.5
        loss = torch.max(loss, weights * r) - weights * r
    return torch.abs(flow_fields[:, 0] + flow_fields[:, 1]).mean()

def full_temporal_smoothness_loss(weights):
    assert(weights.shape[1] == 2)
    past = torch.flip(weights[:, 0], (-1, -2))
    future = weights[:, 1]

    loss = torch.abs(past - future).mean()
    return loss

def spatial_smoothness_loss(weights, image=None, occ_mask=None, edge_weight=1.0):
    # Calculate the gradient along the height and width dimensions
    grad_height = weights[:, :, 1:, :, :, :] - weights[:, :, :-1, :, :, :]
    grad_width = weights[:, :, :, 1:, :, :] - weights[:, :, :, :-1, :, :]

    # Edgeaware
    if image is not None:
        image = image.to(weights.device)
        image_grad_y = image[:, :, :, 1:, :] - image[:, :, :, :-1, :]
        image_grad_x = image[:, :, :, :, 1:] - image[:, :, :, :, :-1]
    else:
        image_grad_y = torch.zeros_like(grad_height)[:, :, None, ..., 0, 0] # reduce by one dim
        image_grad_x = torch.zeros_like(grad_width)[:, :, None, ..., 0, 0]

    if occ_mask is not None:
        occ_mask = occ_mask.to(weights.device)
        #occ_mask_grad_y = occ_mask[:, :, :, 1:, :] - occ_mask[:, :, :, :-1, :]
        #occ_mask_grad_x = occ_mask[:, :, :, :, 1:] - occ_mask[:, :, :, :, :-1]
        occ_mask_grad_y = torch.logical_or(occ_mask[:, :, :, 1:, :], occ_mask[:, :, :, :-1, :])
        occ_mask_grad_x = torch.logical_or(occ_mask[:, :, :, :, 1:], occ_mask[:, :, :, :, :-1])
    else:
        occ_mask_grad_y = torch.zeros_like(grad_height)[:, :, None, ..., 0, 0]
        occ_mask_grad_x = torch.zeros_like(grad_width)[:, :, None, ..., 0, 0]

    # Right now the edgeaware is only on the occ mask edges -- see if this makes the occ mask fill in correctly
    #grad_height = grad_height * torch.exp(-edge_weight * torch.mean(torch.abs(image_grad_y * occ_mask_grad_y), dim=2, keepdim=True))[..., None, None]
    #grad_width = grad_width * torch.exp(-edge_weight * torch.mean(torch.abs(image_grad_x * occ_mask_grad_x), dim=2, keepdim=True))[..., None, None]
    grad_height = grad_height * torch.exp(-edge_weight * torch.mean(torch.abs(image_grad_y), dim=2, keepdim=True))[..., None, None]
    grad_width = grad_width * torch.exp(-edge_weight * torch.mean(torch.abs(image_grad_x), dim=2, keepdim=True))[..., None, None]

    # You can use either the L1 or L2 norm for the gradients.
    # L1 norm (absolute differences)
    loss_height = torch.abs(grad_height).mean()
    loss_width = torch.abs(grad_width).mean()

    # Alternatively, you could use the L2 norm (squared differences)
    # loss_height = (grad_height ** 2).mean()
    # loss_width = (grad_width ** 2).mean()

    # Combine the losses along both dimensions
    loss = loss_height + loss_width

    return loss

def position_spat_smoothness(positions, image=None, edge_weight=1.0):
    # Calculate the gradient along the height and width dimensions
    grad_height = positions[:, :, 1:, :, :] - positions[:, :, :-1, :, :]
    grad_width = positions[:, :, :, 1:, :] - positions[:, :, :, :-1, :]

    # second derivative ? 
    grad_height = grad_height[:, :, 1:, :, :] - grad_height[:, :, :-1, :, :]
    grad_width = grad_width[:, :, :, 1:, :] - grad_width[:, :, :, :-1, :]

    # Edgeaware
    if image is not None:
        image = image.to(weights.device)
        image_grad_y = image[:, :, :, 1:, :] - image[:, :, :, :-1, :]
        image_grad_x = image[:, :, :, :, 1:] - image[:, :, :, :, :-1]
    else:
        image_grad_y = torch.zeros_like(grad_height)[:, :, None, ..., 0, 0] # reduce by one dim
        image_grad_x = torch.zeros_like(grad_width)[:, :, None, ..., 0, 0]
    grad_height = torch.exp(-edge_weight * torch.mean(torch.abs(image_grad_y), dim=2)[..., None])
    grad_width = torch.exp(-edge_weight * torch.mean(torch.abs(image_grad_x), dim=2)[..., None])

    # You can use either the L1 or L2 norm for the gradients.
    # L1 norm (absolute differences)
    loss_height = torch.abs(grad_height).mean()
    loss_width = torch.abs(grad_width).mean()

    # Alternatively, you could use the L2 norm (squared differences)
    # loss_height = (grad_height ** 2).mean()
    # loss_width = (grad_width ** 2).mean()

    # Combine the losses along both dimensions
    loss = loss_height + loss_width

    return loss
def charbonnier(x, alpha=0.5, eps=1e-3):
    return torch.pow(torch.square(x) + eps**2, alpha)

def flat_charbonnier(x, alpha=0.5, eps=1e-3, flat=1e-1):
    x = torch.maximum(torch.abs(x), torch.Tensor([flat]).to(x.device)) - flat
    return charbonnier(x, alpha=alpha, eps=eps)

criterion = lambda x, y, f: torch.nanmean(flat_charbonnier(x - y, flat=f))
criterion_min = lambda x, y, f: torch.mean(torch.min(flat_charbonnier(x-y, flat=f), dim=1, keepdim=True)[0]) # three frame min pixel loss
criterion_three_frames = lambda x, y, f: 0.1 * criterion_min(x, y, f) + 0.9 * criterion(x, y, f) # match pixel values if you can

def normal_minperblob(weights, src, tgt, weight_sl, downsample_factor, filter_zoom, flatness=0.0, use_min=False):
    assert(src.shape[2] == 3)

    if len(src.shape) == 5: # not made into filter-shapes yet
        src_padded = pad_for_filter(src, weight_sl, downsample_factor)

    assert((weight_sl + downsample_factor - 1) % filter_zoom == 0)
    true_sl = (weight_sl + downsample_factor - 1) // filter_zoom
    src_padded = src_padded.reshape((*tuple(src_padded.shape)[:5], true_sl, filter_zoom, true_sl, filter_zoom))
    tgt_padded = torch.broadcast_to(tgt[..., None, None, None, None], src_padded.shape)

    diff = torch.zeros((*tuple(src_padded.shape)[:5], true_sl, true_sl)).to(src.device)
    for i in range(true_sl): # reduce peak memory consumption
        diff_slice = src_padded[..., i, None, :] - tgt_padded[..., i, None, :] # in place version of this has bugs??? ugh (probably these are views or something like that)

        diff_slice = torch.permute(diff_slice, (0, 1, 2, 3, 4, 5, 7, 6, 8))
        diff_slice = torch.reshape(diff_slice, (*tuple(diff_slice.shape)[:7], -1))

        diff_abs = torch.abs(diff_slice) # b n c i j k k2 l l2
        diff_abs = torch.sum(diff_abs, dim=2, keepdim=True)
        _, diff_idx = torch.min(diff_abs, dim=-1)
        diff_idx = torch.broadcast_to(diff_idx, tuple(diff_slice.shape)[:-1])[..., None]

        diff_slice = torch.gather(diff_slice, -1, diff_idx)
        diff[..., i] = diff_slice[..., 0, 0]
    diff = torch.einsum('bnijkl, bncijkl->bncij', weights, diff)

    if src.shape[1] == 2:
        """
        if use_min:
            print('Using min')
            return criterion_three_frames(diff, torch.zeros_like(diff), flatness), _
        else:
            print('Not using min')
            return criterion(diff, torch.zeros_like(diff), flatness), _
        """
        return criterion(diff, torch.zeros_like(diff), flatness), _
    else:
        return criterion(diff, torch.zeros_like(diff), flatness), _

def distribution_photometric(weights, src, tgt, weight_sl, downsample_factor, filter_zoom):
    # image: (N, M, 3, Wbig, Hbig)
    # weights: (N, M, Wsmall, Hsmall, Wfilt, Hfilt)
    assert(src.shape[2] == 3)
    assert(downsample_factor == 1) # until this is perfected

    if len(src.shape) == 5: # not made into filter-shapes yet
        src_padded = pad_for_filter(src, weight_sl * filter_zoom, downsample_factor)
    else:
        src_padded = src

    assert((weight_sl + downsample_factor - 1) % filter_zoom == 0)
    true_sl = (weight_sl + downsample_factor - 1) // filter_zoom
    src_padded = src_padded.reshape((*tuple(src_padded.shape)[:5], true_sl, filter_zoom, true_sl, filter_zoom))
    tgt_padded = torch.broadcast_to(tgt[..., None, None, None, None], src_padded.shape)

    diff = torch.zeros((*tuple(src_padded.shape)[:5], true_sl, true_sl)).to(src.device)
    for i in range(true_sl): # reduce peak memory consumption
        diff_slice = src_padded[..., i, None, :] - tgt_padded[..., i, None, :] # in place version of this has bugs??? ugh (probably these are views or something like that)
        diff_slice = torch.square_(diff_slice)
        diff_slice += 1e-6
        diff_slice = torch.sqrt_(diff_slice)
        diff_slice = torch.sum(diff_slice, dim=2)
        diff_slice = torch.min(diff_slice, dim=5)[0]
        diff_slice = torch.min(diff_slice, dim=6)[0]

        diff[..., i] = diff_slice[..., 0]

    loss = diff * weights
    loss_per_pxl = loss.sum(dim=(4, 5))

    #src_padded[ b n c i j k k2 l l2]
    #b n i j (k) (l)
    # Compute restricted for downstream 
    """
    if filter_zoom > 1:
        # Prep max indices
        weights_, x_ind_ = torch.max(weights, dim=-1, keepdim=True)
        _, y_ind = torch.max(weights_, dim=-2, keepdim=True)
        x_ind = torch.gather(x_ind_, -2, y_ind)
        mode_idx = y_ind * weight_sl + x_ind

        # Reshape src for indexing
        src_padded = src_padded.permute((0, 1, 2, 3, 4, 5, 7, 6, 8))
        src_padded = src_padded.reshape((*tuple(src_padded.shape)[:5], weight_sl ** 2, filter_zoom, filter_zoom))

        # Align shapes
        mode_idx = mode_idx[..., 0]
        mode_idx = torch.broadcast_to(mode_idx[:, :, None, :, :, :, None, None], src_padded.shape)
        mode_idx = mode_idx[..., 0, None, :, :]

        # Gather
        small_src = torch.gather(src_padded, 5, mode_idx)[..., 0, :, :]
    else:
        small_src = None
    """
    small_src = None
    return torch.nanmean(loss_per_pxl), small_src

def bijectivity_loss(outputs, downsample_factor=1):
    transposed = transpose_filter(torch.flip(outputs["weights"], dims=[1]), downsample_factor=downsample_factor)
    downsampled = downsample_filter(outputs["weights"], downsample_factor=downsample_factor)
    matched = torch.sum(torch.sqrt(transposed * downsampled), dim=(-2, -1))
    weight_sl = downsampled.shape[-3]

    return torch.mean(1 - matched) * (weight_sl ** 2)
