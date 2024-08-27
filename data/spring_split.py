import random
from glob import glob

def get_scene(s, base_path):
    s = s[len(base_path):]
    s = s[:s.index("/")]
    return s

def frame_to_fwd_flow(s):
    s = s.replace("frame_", "flow_FW_")
    s = s.replace(".png", ".npy")
    return s

base_path = "/data/scene-rep/custom/spring/train/" # replace with your path
split_path = "splits/spring/spring_split.txt"

files = list(glob(base_path + "*/frame_left/frame_left_*.png"))
files = list(sorted(files))
with open(split_path, "w") as f:
    for a, b, c in zip(files[:-2],files[1:-1],files[2:]):
        a_scene, b_scene, c_scene = get_scene(a, base_path), get_scene(b, base_path), get_scene(c, base_path)
        if a_scene == b_scene and b_scene == c_scene:
            if random.random() < 0.2:
                f.write(b + ",validation\n")
            else:
                f.write(b + ",training\n")

print('Spring splits generated!')
