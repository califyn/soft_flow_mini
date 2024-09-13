import argparse
import time
import subprocess

job = """#!/bin/bash

#SBATCH -J soft_flow_{name}
#SBATCH -o /data/scratch/califyn/flow_mini/slurm_logs/{name}_%j.out
#SBATCH -e /data/scratch/califyn/flow_mini/slurm_logs/{name}_%j.out
#SBATCH --mail-user='califyn@mit.edu'
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:{type_gpus}{num_gpus}
#SBATCH --time={hours}:0:0
#SBATCH --mem={memory}
#SBATCH --partition=vision-sitzmann
#SBATCH --qos=vision-sitzmann-low
#SBATCH --cpus-per-task=32
#SBATCH --requeue

rm /data/scratch/califyn/flow_mini/slurm_logs/latest.out
ln -s "/data/scratch/califyn/flow_mini/slurm_logs/{name}_$SLURM_JOB_ID.out" /data/scratch/califyn/flow_mini/slurm_logs/latest.out

source /data/scratch/califyn/.bashrc
conda activate croco
cd /data/scratch/califyn/flow_mini
ulimit -n 50000
echo "About to execute..."
stdbuf -o0 -e0 srun python main.py -n {name} -m {mode} -c {config} -d {id} -r {resume}
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--name', required=True)
    parser.add_argument('-m','--mode', required=True)
    parser.add_argument('-c','--config', default="overfit")

    parser.add_argument('-t','--type_gpus', default="a100")
    parser.add_argument('-g','--num_gpus', default="1")
    parser.add_argument('-r','--hours', default="96")
    parser.add_argument('-e','--memory', default="16G")

    parser.add_argument('-d','--id', default=None)
    parser.add_argument('-s','--resume', default="allow")

    args = parser.parse_args()

    job = job.replace("{name}", args.name)
    job = job.replace("{mode}", args.mode)
    job = job.replace("{config}", args.config)

    if args.type_gpus == "any":
        args.type_gpus = ""
    else:
        args.type_gpus += ":"
    job = job.replace("{type_gpus}", args.type_gpus)
    job = job.replace("{num_gpus}", args.num_gpus)
    job = job.replace("{memory}", args.memory)
    job = job.replace("{hours}", args.hours)

    if args.id is None:
        new_id = str(time.time_ns())[-16:]
        job = job.replace("{id}", new_id)
    else:
        job = job.replace("{id}", args.id)
    job = job.replace("{resume}", args.resume)

    with open("job.sh", "w") as f:
        f.write(job)

    input(job + "\nContinue?")
    subprocess.run("sbatch job.sh", shell=True)
