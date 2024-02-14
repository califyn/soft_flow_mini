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

def spatial_smoothness_loss(weights):
    # Calculate the gradient along the height and width dimensions
    grad_height = weights[:, :, 1:, :, :, :] - weights[:, :, :-1, :, :, :]
    grad_width = weights[:, :, :, 1:, :, :] - weights[:, :, :, :-1, :, :]

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

criterion = lambda x, y: torch.mean(charbonnier(x - y))
#criterion = lambda x, y: torch.mean(torch.min(charbonnier(x-y), dim=1, keepdim=True)[0]) # three frame min pixel loss
