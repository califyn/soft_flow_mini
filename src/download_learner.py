import pathlib
import torch
from omegaconf import OmegaConf

from wandb_utils import download_latest_checkpoint, rewrite_checkpoint_for_compatibility
from .learner import OverfitSoftLearner
from ..data.sintel_superres import SintelSuperResDataset
from .soft_utils, .soft_losses import *

# Some example code showing how to play with a trained soft flow
# Note: this creates a run, i don't know why

# Get model
run_id = "scene-representation-group/soft_flow/gix51zyz"
path = download_latest_checkpoint(run_id, pathlib.Path(__file__).parent.resolve() / "downloads")
checkpoint = torch.load(path)

cfg = OmegaConf.load('overfit.yaml')
model = OverfitSoftLearner(cfg)
model.load_state_dict(checkpoint['state_dict'])

model.cuda()

# Get training example
dataset = SintelSuperResDataset(cfg)
batch = dataset[0]
batch = [tensor.cuda() for tensor in batch]

# Do whatever you like ...
print(batch, model)
pass
