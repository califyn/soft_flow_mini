# Soft flow mini repository

Includes code to run some soft flow things. This git repository automatically uploads to weights and biases at the ``soft_flow`` project (can be changed in ``overfit.yaml``).

To train a model, do
```
python main.py --name name_of_run --mode {online,dryrun,offline,disabled}
```
Note: pytorch lightning will try to use all the gpus on the machine.

## Files
- ``main.py``: training loop, shouldn't need to be modified
- ``overfit_soft_learner.py``: code for the soft flow model. This version uses superresolution, warping a 1024x512 source frame (frame 3) to a 256x128 target frame (frame 2), learning flows frame2 to frame3 (i think). Includes brief explanations of the logged data
- ``sintel_superres.py``: dataloading. Note: the path to the Sintel dataset needs to be changed
- ``roaming_images.py``: roamingimages dataset, toy dataset. Not integrated with the current soft flow learner
- ``overfit.yaml``: configuration file
- ``download_learner.py``: some minimal code to download trained models (models are currently saved every 5000 steps), and then you can run arbitrary code on them
- ``soft_losses.py, soft_utils.py``: losses and utilities for soft flow
