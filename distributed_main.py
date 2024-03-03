import sys
from pathlib import Path
import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sintel_superres import SintelSuperResDataset
from distributed_soft_learner import OverfitSoftLearner

import wandb
from wandb_utils import download_latest_checkpoint, rewrite_checkpoint_for_compatibility
from omegaconf import DictConfig, OmegaConf

import lightning as L
import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

def run(rank, world_size, name, mode):
    print(f'Running {rank} out of {world_size}...')
    cfg = OmegaConf.load('distributed_overfit.yaml')

    cfg.wandb.name = name
    cfg.wandb.mode = mode if rank == 0 else "disabled"

    # Set up dataset and dataloader
    dataset = SintelSuperResDataset(cfg)
    train_dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=cfg.training.data.batch_size,
                num_workers=0,
                shuffle=False
            )
    val_dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=cfg.training.data.batch_size,
                num_workers=0,
                shuffle=False
            )

    # Create model
    model = OverfitSoftLearner(cfg, rank, world_size)

    # Enforce the correct Python version.
    if sys.version_info.major < 3 or sys.version_info.minor < 9:
        print(
            "Please use Python 3.9+. If on IBM Satori, "
            "install Anaconda3-2022.10-Linux-ppc64le.sh"
        )

    # Set up logging with wandb.
    if cfg.wandb.mode != "disabled":
        # If resuming, merge into the existing run on wandb.
        resume_id = cfg.wandb.get("resume", None)
        logger = WandbLogger(
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            name=cfg.wandb.name,
            log_model="all",
            config=OmegaConf.to_container(cfg),
            id=None if cfg.wandb.get("use_new_id", False) else resume_id,
        )

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            print(f"wandb mode: {wandb.run.settings.mode}")
            wandb.run.log_code(".")
    else:
        resume_id = None
        logger = None

    # If resuming a run, download the checkpoint.
    if resume_id is not None:
        run_path = f"{cfg.wandb.entity}/{cfg.wandb.project}/{resume_id}"
        print(run_path)
        checkpoint_path = download_latest_checkpoint(
            run_path, Path("outputs/loaded_checkpoints")
        )
        checkpoint_path = rewrite_checkpoint_for_compatibility(checkpoint_path)
    else:
        checkpoint_path = None

    callbacks = [
                LearningRateMonitor("step", True),
                ModelCheckpoint(every_n_train_steps=5000)
            ]

    # Initialize Pytorch Lightning trainer
    trainer = L.Trainer(
        max_epochs=-1,
        accelerator='auto',
        logger=logger,
        devices=[rank],
        callbacks=callbacks,
        precision=cfg.training.precision,
        check_val_every_n_epoch=cfg.validation.check_epoch,
        val_check_interval=cfg.validation.check_interval,
        overfit_batches=None if "overfit_batch" not in dir(cfg.training) else cfg.training.overfit_batch,
        limit_val_batches=None if "limit_batch" not in dir(cfg.validation) else cfg.validation.limit_batch
    )

    # Training happens here
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=checkpoint_path
    )

def init_process(rank, size, fn, name, mode, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, name, mode)

if __name__ == "__main__":
    size = 5
    processes = []
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--name', required=True)
    parser.add_argument('-m','--mode', required=True)
    args = parser.parse_args()

    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run, args.name, args.mode))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
