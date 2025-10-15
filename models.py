import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead

class GaussianNoise(nn.Module):
    """Applies Gaussian noise to the input tensor."""
    def __init__(self, stddev=0.15):
        super().__init__()
        self.stddev = stddev

    def forward(self, x):
        noise = torch.randn_like(x) * self.stddev
        return x + noise

class AllConvNet(nn.Module):
    """
    All convolutional neural network with Gaussian noise added to each layer.
    The number of layers (7) is chosen to convolve an input image of size 128x128
    down to a 1D feature map.

    Parameters
    ----------
    layer_dim : int
        Number of output channels for each convolutional layer.
    noise_std : float
        Standard deviation of the Gaussian noise added to each layer.
    """
    def __init__(self, layer_dim=96, noise_std=0.15):
        super(AllConvNet, self).__init__()
        self.layer_dim = layer_dim
        
        self.conv1 = nn.Conv2d(3, layer_dim, kernel_size=3, stride=1, padding=1)
        self.noise1 = GaussianNoise(noise_std)
        
        self.conv2 = nn.Conv2d(layer_dim, layer_dim, kernel_size=3, stride=2, padding=1)
        self.noise2 = GaussianNoise(noise_std)
        
        self.conv3 = nn.Conv2d(layer_dim, layer_dim, kernel_size=3, stride=2, padding=1)
        self.noise3 = GaussianNoise(noise_std)

        self.conv4 = nn.Conv2d(layer_dim, layer_dim, kernel_size=3, stride=2, padding=1)
        self.noise4 = GaussianNoise(noise_std)

        self.conv5 = nn.Conv2d(layer_dim, layer_dim, kernel_size=3, stride=2, padding=1)
        self.noise5 = GaussianNoise(noise_std)

        self.conv6 = nn.Conv2d(layer_dim, layer_dim, kernel_size=3, stride=2, padding=1)
        self.noise6 = GaussianNoise(noise_std)

        self.conv7 = nn.Conv2d(layer_dim, layer_dim, kernel_size=3, stride=2, padding=1)
        self.noise7 = GaussianNoise(noise_std)

    def forward(self, x):
        x = F.relu(self.noise1(self.conv1(x)))
        x = F.relu(self.noise2(self.conv2(x)))
        x = F.relu(self.noise3(self.conv3(x)))
        x = F.relu(self.noise4(self.conv4(x)))
        x = F.relu(self.noise5(self.conv5(x)))
        x = F.relu(self.noise6(self.conv6(x)))
        x = self.noise7(self.conv7(x))
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = x.view(x.size(0), -1)
        return x

class SimCLR(pl.LightningModule):
    """
    Use SimCLR framework for self-supervised learning of image representations on ImageNet using an AllConvNet backbone.

    Parameters
    ----------
    layer_dim : int
        Number of output channels for each convolutional layer in the AllConvNet.
    noise_std : float
        Standard deviation of the Gaussian noise added to each layer in the AllConvNet.
    lr : float
        Learning rate for the SGD optimizer.
    momentum : float
        Momentum for the SGD optimizer.
    weight_decay : float
        Weight decay (L2 regularization) for the SGD optimizer.
    max_epochs : int
        Maximum number of training epochs (used for learning rate scheduling).
    """
    def __init__(self, layer_dim=96, noise_std=0.15, lr=6e-2, momentum=0.9, weight_decay=5e-4, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = AllConvNet(layer_dim, noise_std)
        self.projection_head = SimCLRProjectionHead(layer_dim, layer_dim, 128)
        self.criterion = NTXentLoss()

    def forward(self, x):
        h = self.backbone(x)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.hparams.max_epochs)
        return [optim], [scheduler]