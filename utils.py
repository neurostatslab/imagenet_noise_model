import os
import glob
import torch
import torchvision
from lightly.data import LightlyDataset
from lightly.transforms import SimCLRTransform, utils
import pytorch_lightning as pl
import warnings
from models import SimCLR

def imagenet_datasets(imagenet_path, input_size=128):
    """
    Initialize imagenet dataset for training ("train" folder) and testing ("val" folder)

    Parameters
    ----------
    imagenet_path : str
        Path to the ImageNet dataset. The path should contain two folders 'train' and 'val'.
    input_size : int, optional
        Input size of the images. Default is 224.

    Returns
    -------
    dataset_train : lightly.data.Dataset
        Dataset for the training images.
    dataloader_test : lightly.data.Dataset
        Dataset for the test images.
    """
    # SimCLR augmentations for training
    train_transform = SimCLRTransform(input_size=input_size)
    dataset_train = LightlyDataset(input_dir=os.path.join(imagenet_path,"train"), transform=train_transform)

    # image resizing and normalization for testing
    test_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((input_size, input_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=utils.IMAGENET_NORMALIZE["mean"],
                std=utils.IMAGENET_NORMALIZE["std"],
            ),
        ]
    )
    dataset_test = LightlyDataset(input_dir=os.path.join(imagenet_path,"val"), transform=test_transform)
    return dataset_train, dataset_test
    

def imagenet_dataloaders(imagenet_path, input_size=128, batch_size=128, num_workers=8, subsample=2):
    """
    Make train and test dataloaders for ImageNet dataset.

    Parameters
    ----------
    imagenet_path : str
        Path to the ImageNet dataset. The path should contain two folders 'train' and 'val'.
    input_size : int, optional
        Input size of the images. Default is 224.
    batch_size : int, optional
        Batch size for the dataloaders. Default is 128.
    num_workers : int, optional
        Number of workers for the dataloaders. Default is 8.
    subsample : int, optional
        Subsample the dataset by this factor. Default is 2 (use every second image).

    Returns
    -------
    dataloader_train : torch.utils.data.DataLoader
        Dataloader for the training set.
    dataloader_test : torch.utils.data.DataLoader
        Dataloader for the test set.
    """
    # get datasets
    dataset_train, dataset_test = imagenet_datasets(imagenet_path, input_size)

    # subsample if requested
    if subsample > 1:
        subset = list(range(0, len(dataset_train), subsample))
        dataset_train = torch.utils.data.Subset(dataset_train, subset)
        subset = list(range(0, len(dataset_test), 2))
        dataset_test = torch.utils.data.Subset(dataset_test, subset)
        
    # make loaders
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )   
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    return dataloader_train, dataloader_test

def load_imagenet_model(checkpoint_path, checkpoint_version=None, return_checkpoint=False):
    """
    Load an ImageNet SimCLR model from a checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint directory. Should contain a folder 'lightning_logs' with version subfolders.
    checkpoint_version : int, optional
        Version of the checkpoint to load. If None, the latest version is loaded. Default is None.
    return_checkpoint : bool, optional
        Whether to return the path from where the checkpoint was loaded.

    Returns
    -------
    model : SimCLR
        The loaded SimCLR model.
    checkpoint : str
        Path to the loaded checkpoint file (used if model training is being resumed from checkpoint).
    """
    if checkpoint_version is None:
        version = sorted(glob.glob(os.path.join(checkpoint_path,"lightning_logs","*")), key=os.path.getmtime)[-1]
        checkpoint = sorted(glob.glob(os.path.join(version,"checkpoints","*.ckpt")), key=os.path.getmtime)[-1]
    else:
        checkpoint = sorted(glob.glob(os.path.join(checkpoint_path,"lightning_logs","version_"+str(checkpoint_version),"checkpoints","*.ckpt")), key=os.path.getmtime)[-1]

    if return_checkpoint:
        return SimCLR.load_from_checkpoint(checkpoint), checkpoint
    else:
        return SimCLR.load_from_checkpoint(checkpoint)

def train_imagenet_model(dataloader, checkpoint_path, load_from_checkpoint=True, checkpoint_version=None, max_time=None, max_epochs=100, **kwargs):
    """
    Train an ImageNet SimCLR model using an unlerlying CNN with noise injected at each layer.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Dataloader for the training set. Get it from imagenet_data_loaders().
    checkpoint_path : str
        Where to save (and/or load) the model checkpoints.
    load_from_checkpoint : bool, optional
        Whether to load the model from a checkpoint if it exists. Default is True.
    checkpoint_version : int, optional
        Version of the checkpoint to load. If None, the latest version is loaded. Default is None.
    max_epochs : int, optional
        Maximum number of epochs to train the model. Default is 100.

    Returns
    -------
    model : SimCLR
        The trained SimCLR model.
    """
    trainer = pl.Trainer(max_epochs=max_epochs, max_time=max_time, devices=1, accelerator="gpu", default_root_dir=checkpoint_path)

    if load_from_checkpoint:
        try:
            model, checkpoint = load_imagenet_model(checkpoint_path, checkpoint_version, return_checkpoint=True)
        except:
            warnings.warn("No model checkpoint exists at specified path. Running with a new model.")
            model = SimCLR(**kwargs)
            checkpoint = None
        
    else:
        model = SimCLR(**kwargs)
        checkpoint = None

    trainer.fit(model, dataloader, ckpt_path=checkpoint)    
    return model