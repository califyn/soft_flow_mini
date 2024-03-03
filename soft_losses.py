import torch.distributions as dist
import torch
from soft_utils import warp_previous_flow

def entropy_loss(weight_softmax):
    entropy = -torch.mean(weight_softmax * torch.log(weight_softmax + 1e-8))
    return entropy

def temporal_smoothness_loss(flow_fields):
    expected_flow_t = warp_previous_flow(flow_fields)
    # expected_flow_t = warp_previous_flow_multi(flow_fields)

    flow_t_plus_1 = flow_fields[:, 1:].reshape(-1, *flow_fields.shape[2:])
    flow_diffs = flow_t_plus_1 - expected_flow_t
    loss = (flow_diffs ** 2).mean()
    return loss

def spatial_smoothness_loss(weights, image=None, occ_mask=None, edge_weight=1, occ_weight=0.0):
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
        occ_mask_grad_y = occ_mask[:, :, :, 1:, :] - occ_mask[:, :, :, :-1, :]
        occ_mask_grad_x = occ_mask[:, :, :, :, 1:] - occ_mask[:, :, :, :, :-1]
    else:
        occ_mask_grad_y = torch.zeros_like(grad_height)[:, :, None, ..., 0, 0]
        occ_mask_grad_x = torch.zeros_like(grad_width)[:, :, None, ..., 0, 0]

    grad_height = grad_height * torch.exp(-edge_weight * torch.mean(torch.abs(image_grad_y - occ_weight * occ_mask_grad_y), dim=2, keepdim=True))[..., None, None]
    grad_width = grad_width * torch.exp(-edge_weight * torch.mean(torch.abs(image_grad_x - occ_weight * occ_mask_grad_x), dim=2, keepdim=True))[..., None, None]

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

def position_spat_smoothness(positions):
    # Calculate the gradient along the height and width dimensions
    grad_height = positions[:, :, 1:, :, :] - positions[:, :, :-1, :, :]
    grad_width = positions[:, :, :, 1:, :] - positions[:, :, :, :-1, :,]

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

criterion = lambda x, y: torch.nanmean(charbonnier(x - y))
criterion_min = lambda x, y: torch.mean(torch.min(charbonnier(x-y), dim=1, keepdim=True)[0]) # three frame min pixel loss
criterion_three_frames = lambda x, y: criterion_min(x, y) + 5e-2 * criterion(x, y) # match pixel values if you can
