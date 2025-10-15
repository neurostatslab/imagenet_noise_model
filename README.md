# imagenet_noise_model

This small repository gives a brief demo on how to train and analyze a deep convolutional neural network (CNN) on images from the ImageNet dataset, where Gaussian noise is injected at each layer of the CNN to simulate noise in neural systems. This repository contains code for training the network on the Flatiron cluster either in a jupyter notebook (`train_imagenet_model.ipynb`) or as a slurm job (`train_imagenet_model.sh`) using a GPU.

We use the SimCLR ([Chen et al. 2020](https://arxiv.org/abs/2002.05709)) framework to train the network via *self-supervised learning*, where the model is trained to minimize the distance between two different, random augmentations of the same image. See [this](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html) tutorial for a more in-depth explanation of SimCLR.

Briefly, this training framework consists of:  
1.  A data aumentation module that transforms any input image with a series of random operations (e.g. cropping, resizing, blurring), to create two correlated views.
2. A base encoder or "backbone" network to extract feature representations from the input images (in this case, the CNN with Gaussian noise)
3. A projection head, or a small neural network that maps feature representations into a space in which the loss function is applied, which had been shown to improve model accuracy.
4. A contrastive loss funciton, which seeks to maximize agreement between representations of separate augmentations of the same image. 

For implementations of (1), (2), and (4), we will use the package `lightly`. 

## Core scripts
- `models.py`: contains the pytorch models for the CNN with Gaussian noise and for training with SimCLR
- `train.py`: script for training the model, listing out settable parameters and default values, and pulling them from the environment variables for training on the cluster
- `utils.py`: helper functions for loading ImageNet datasets and dataloaders, as well as helper functions for training and loading models.

## Installation
After connecting to the Flatiron cluster, create and activate a virtual environment with `python`, `cuda`, and `cudnn` loaded as follows:
```bash
module load python cuda cudnn
python -m venv --system-site-packages ~/venvs/imagenet
```
This will create the virtual environment `imagenet` in the `venvs` subdrectory of your home folder. The `--system-site-packages` flag will include most of the packages we need for training the model except for `lightly`, which needs to be installed after activating the environment.
```bash
source ~/venvs/imagenet/bin/activate
pip install lightly
```
If you want to use this environment in a jupyter notebook, install it as a jupyter kernel using the following command
```bash
python -m ipykernel install --user --name imagenet
```
Note that this needs to be done with `cuda` loaded in order to make the jupyter kernal CUDA-aware.

Clone this repository somewhere in your cluster home folder:
```
git clone https://github.com/neurostatslab/imagenet_noise_model.git
```

## How to use
There are two ways to train the model, either interactively in a jupyter notebook or automated through a slurm job. In both cases, the ImageNet dataset will need to be copied to the local storage of the requested node. This is automated during Slurm jobs, but will need to be done manually for the jupyter server.

### Slurm
While inside the `imagenet_noise_model` folder, you can submit a job on the cluster using the bash file `train_imagenet_model.sh` as follows:
```
sbatch train_imagenet_model.sh 0.15 /mnt/ceph/users/svenditto/imgnet-models/noise015
```
where the first value after `train_imagenet_model.sh` is the Gaussian noise standard deviation to use, and the second value is the path to save the model checkpoints. 

Under the hood, this file call the script `train.py` to pull any exported environment variables and train the model. This file can be modified to adjust parameters of the Slurm job

### Jupyter notebook
The simplest way to get started in a jupyter notebook on the cluster is to request a server through [JupyterHub](https://jupyter.flatironinstitute.org/). You'll want a GPU node with at least 8 CPU cores and 12GB ram.
