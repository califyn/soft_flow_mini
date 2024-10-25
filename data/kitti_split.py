from superres import load_flow
import numpy as np

import random
from glob import glob

def frame_to_fwd_flow(s):
    return s.replace("image_2", "flow_occ")

base_path = f"/data/scene-rep/custom/KITTI/KITTI/training/" # replace with your path
split_path = "splits/kitti/kitti_split.py"

frameskip = 1
files = [
    list(glob(base_path + f"image_2/*_{(10-frameskip):02d}.png")),
    list(glob(base_path + "image_2/*_10.png")),
    list(glob(base_path + f"image_2/*_{(10+frameskip):02d}.png")),
]
files = [list(sorted(files_sub)) for files_sub in files]
with open(split_path, "w") as f:
    for a, b, c in zip(*files):
        if random.random() < 0.2:
            f.write(b + ",validation\n")
        else:
            f.write(b + ",training\n")

flow_max_dict = {}
for file in files[1]:
    flow_np = load_flow(frame_to_fwd_flow(file))
    flow_max_dict[file] = np.nanmax(np.abs(flow_np))

with open("splits/kitti/kitti_max.txt", "w") as f:
    for k, v in flow_max_dict.items():
        f.write(k + "," + str(v) + "\n")

print('KITTI splits generated!')
