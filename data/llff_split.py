import numpy as np
from tqdm import tqdm

import random, math
from glob import glob

from torchvision.models.optical_flow import raft_large
from torchvision.transforms.functional import to_tensor, resize
import cv2

def do_raft_prediction(frame1, frame2, smurf=False):
    if not smurf:
        model = raft_large(weights='C_T_V2')
    else:
        model = raft_smurf(checkpoint="smurf_sintel.pt")
    model.cuda()
    
    mem_limit_scale = 1 / max(frame1.shape[2] / 1024, frame1.shape[3] / 1024, 1) # scale down for memory limits
    def scale_dim_for_memory(x):
        return [8 * math.ceil(xx / 8 * mem_limit_scale) for xx in x]
    frame1_round = resize(frame1, scale_dim_for_memory(frame1.shape[2:]))
    frame2_round = resize(frame2, scale_dim_for_memory(frame2.shape[2:]))
    
    return model(frame1_round, frame2_round)[-1] / mem_limit_scale

base_path = f"/nobackup/nvme1/datasets/llff/*/images/" # replace with your path
split_path = "splits/llff/llff_split.py"

frameskip = 1
files = list(glob(base_path + "*.jpg")) + list(glob(base_path + "*.JPG"))
files = list(sorted(files))
files = [[a, b, c] for a, b, c in zip(files[:-2], files[1:-1], files[2:])]

with open(split_path, "w") as f:
    for a, b, c in files:
        if random.random() < 0.2:
            f.write(b + ",validation\n")
        else:
            f.write(b + ",training\n")

print('Starting RAFT guesses for max flow...')
flow_max_dict = {}
for seq in tqdm(files):
    frame0 = cv2.cvtColor(cv2.imread(seq[0]), cv2.COLOR_BGR2RGB)
    frame1 = cv2.cvtColor(cv2.imread(seq[1]), cv2.COLOR_BGR2RGB)
    frame2 = cv2.cvtColor(cv2.imread(seq[2]), cv2.COLOR_BGR2RGB)
    frame0 = to_tensor(frame0)[None].cuda()
    frame1 = to_tensor(frame1)[None].cuda()
    frame2 = to_tensor(frame2)[None].cuda()

    flow_pred_2 = do_raft_prediction(frame1, frame2).cpu().detach().numpy()
    flow_pred_1 = do_raft_prediction(frame1, frame0).cpu().detach().numpy()
    flow_max_dict[seq[1]] = max(np.nanmax(np.abs(flow_pred_2)), np.nanmax(np.abs(flow_pred_1)))

with open("splits/llff/llff_max.txt", "w") as f:
    for k, v in flow_max_dict.items():
        f.write(k + "," + str(v) + "\n")

print('LLFF splits generated!')
