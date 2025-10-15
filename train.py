
import os
from utils import imagenet_dataloaders, train_imagenet_model

# the copied and extracted imagenet path should always be here
imagenet_path = "/tmp/imagenet/"

### get environment variables that can be set in the slurm script

## data io parameters
# where to save model checkpoints. should be in your ceph folder
checkpoint_path = os.getenv("CHECKPOINT_PATH", None)
if checkpoint_path is None:
    raise RuntimeError("CHECKPOINT_PATH environment variable not set. It should be set before running this script.")
elif not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path) 
# which version of the checkpoint to load. if None, the latest version is loaded
checkpoint_version = os.getenv("CHECKPOINT_VERSION", None) 
# whether to load the model from a checkpoint if it exists
load_from_checkpoint = bool(int(os.getenv("LOAD_FROM_CHECKPOINT", 0))) 
# input image size
input_size = int(os.getenv("INPUT_SIZE", 128)) 
# batch size
batch_size = int(os.getenv("BATCH_SIZE", 256)) 
# subsample the dataset by this factor (e.g. 2 means use every other image)
subsample = int(os.getenv("SUBSAMPLE", 2)) 
# number of workers for data loading
num_workers = int(os.getenv("NUM_WORKERS", 8)) 
# maximum number of epochs to train
max_epochs = int(os.getenv("MAX_EPOCHS", 200)) 
# maximum time to train the model (e.g. "00:06:00:00" for 6 hours). can set to job time limit (or a little less) for safer(?) stopping.
max_time = os.getenv("MAX_TIME", None) 

## model parameters
# standard deviation of the Gaussian noise added to each layer in the AllConvNet
noise_std = float(os.getenv("NOISE_STD", 0.15)) 
# number of output channels for each convolutional layer in the AllConvNet
layer_dim = int(os.getenv("LAYER_DIM", 96)) 
# SGD learning rate
lr = float(os.getenv("LEARNING_RATE", 6e-2)) 
# SGD momentum
momentum = float(os.getenv("MOMENTUM", 0.9)) 
# SGD weight decay (L2 regularization)
weight_decay = float(os.getenv("WEIGHT_DECAY", 5e-4)) 

# print out the parameters
print("imagenet_path       :", imagenet_path)
print("checkpoint_path     :", checkpoint_path)
print("checkpoint_version  :", checkpoint_version)
print("load_from_checkpoint:", load_from_checkpoint)
print("input_size          :", input_size)
print("batch_size          :", batch_size)
print("subsample           :", subsample)
print("num_workers         :", num_workers)
print("max_epochs          :", max_epochs)
print("max_time            :", max_time)
print("noise_std           :", noise_std)
print("layer_dim           :", layer_dim)
print("learning_rate       :", lr)
print("momentum            :", momentum)
print("weight_decay        :", weight_decay)

# get data loaders
dataloader_train, dataloader_test = imagenet_dataloaders(imagenet_path,
    input_size=input_size, 
    batch_size=batch_size, 
    subsample=subsample, 
    num_workers=num_workers
)
# train the model
model = train_imagenet_model(dataloader_train, checkpoint_path, 
    load_from_checkpoint=load_from_checkpoint, 
    checkpoint_version=checkpoint_version, 
    max_time=max_time, 
    max_epochs=max_epochs,
    layer_dim=layer_dim,
    noise_std=noise_std,
    lr=lr,
    momentum=momentum,
    weight_decay=weight_decay
)





