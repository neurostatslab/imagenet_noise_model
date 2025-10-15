#!/bin/bash

#SBATCH --job-name=imagenet_noise_model
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=12GB
#SBATCH --time=24:00:00

# copy and unzip imagenet data to local /tmp
cp -r /mnt/home/gkrawezik/ceph/AI_DATASETS/ImageNet/2012/imagenet /tmp/imagenet/
pushd /tmp/imagenet/
mkdir val train
cd train/
unzip -qq ../train.zip 
cd ../val
unzip -qq ../val.zip
popd

# load modules and activate virtual environment
module load python cuda cudnn
source /mnt/home/svenditto/venvs/imgnet-gpu/bin/activate

# set environment variables for training script
export NOISE_STD=$1
export CHECKPOINT_PATH=$2
export MAX_TIME=00:23:00:00
python train.py
