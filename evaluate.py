import torchvision
from src.evaluation import *
import subprocess

#subprocess.run("python RAFT/evaluate.py --model=models/raft-kitti.pth --dataset=kitti --mixed_precision", shell=True)

#print(validate_sintel())
#out = run_raft_eval()
#out = run_smurf_eval()
out = run_smurf_against_raft()
#out = run_overfit_eval()
#out = run_overfit_against_smurf()

print({k: v.nanmean().item() for k, v in out['metrics'].items()})
if 'metrics2' in out:
    print({k: v.nanmean().item() for k, v in out['metrics2'].items()})
torchvision.utils.save_image(out['random'], 'logs/random.png')
torchvision.utils.save_image(out['worst'], 'logs/worst.png')
torchvision.utils.save_image(out['best'], 'logs/best.png')
if 'worst_comp' in out:
    torchvision.utils.save_image(out['worst_comp'], 'logs/worst_comp.png')
    torchvision.utils.save_image(out['best_comp'], 'logs/best_comp.png')
