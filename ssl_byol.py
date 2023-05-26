import random
from typing import Callable, Tuple
from kornia import augmentation as aug
from kornia import filters
from kornia.geometry import transform as tf
import torch
from torch import nn, Tensor

from typing import Union

from copy import deepcopy
from itertools import chain
from typing import Dict, List
import pytorch_lightning as pl
from torch import optim
import torch.nn.functional as f

from os import cpu_count

from torch.utils.data import DataLoader
from torchvision.models import resnet18

######MINE
import medmnist
from medmnist import INFO

import os
import argparse
import multiprocessing
from pathlib import Path
from PIL import Image
import numpy as np

from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader


######### MY CONSTS
BATCH_SIZE = 32   #I had 32 they may need 128
IMAGE_SIZE = 28 
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
NUM_WORKERS = multiprocessing.cpu_count()
print("NUM_WORKERS: ", NUM_WORKERS)

############### DATASETS - C# load the data via my method from pneumoniamnist.py
from torchvision.transforms import ToTensor

data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert to RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize for RGB
])

data_flag = 'breastmnist'
info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])
TRAIN_DATASET = DataClass(split='train', transform=data_transform, download=False)
TEST_DATASET = DataClass(split='test', transform=data_transform, download=False)

# encapsulate data into dataloader form
train_loader = DataLoader(dataset=TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=TRAIN_DATASET, batch_size=2*BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset=TEST_DATASET, batch_size=2*BATCH_SIZE, shuffle=False)




############Augment

class RandomApply(nn.Module):
    def __init__(self, fn: Callable, p: float):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return x if random.random() > self.p else self.fn(x)


def default_augmentation(image_size: Tuple[int, int] = (28, 28)) -> nn.Module:
    return nn.Sequential(
        tf.Resize(size=image_size),
        RandomApply(aug.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8),
        aug.RandomGrayscale(p=0.2),
        aug.RandomHorizontalFlip(),
        RandomApply(filters.GaussianBlur2d((3, 3), (1.5, 1.5)), p=0.1),
        aug.RandomResizedCrop(size=image_size),
        aug.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225]),
        ),
    )




##########################Encoder WRapper



def mlp(dim: int, projection_size: int = 256, hidden_size: int = 4096) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size),
    )


class EncoderWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        projection_size: int = 256,
        hidden_size: int = 4096,
        layer: Union[str, int] = -2,
    ):
        super().__init__()
        self.model = model
        self.projection_size = projection_size
        self.hidden_size = hidden_size
        self.layer = layer

        self._projector = None
        self._projector_dim = None
        self._encoded = torch.empty(0)
        self._register_hook()

    @property
    def projector(self):
        if self._projector is None:
            self._projector = mlp(
                self._projector_dim, self.projection_size, self.hidden_size
            )
        return self._projector

    def _hook(self, _, __, output):
        output = output.flatten(start_dim=1)
        if self._projector_dim is None:
            self._projector_dim = output.shape[-1]
        self._encoded = self.projector(output)

    def _register_hook(self):
        if isinstance(self.layer, str):
            layer = dict([*self.model.named_modules()])[self.layer]
        else:
            layer = list(self.model.children())[self.layer]

        layer.register_forward_hook(self._hook)

    def forward(self, x: Tensor) -> Tensor:
        _ = self.model(x)
        return self._encoded    




######################## BYOL n' TRAIN

def normalized_mse(x: Tensor, y: Tensor) -> Tensor:
    x = f.normalize(x, dim=-1)
    y = f.normalize(y, dim=-1)
    return 2 - 2 * (x * y).sum(dim=-1)


class BYOL(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        image_size: Tuple[int, int] = (28, 28),
        hidden_layer: Union[str, int] = -2,
        projection_size: int = 256,
        hidden_size: int = 4096,
        augment_fn: Callable = None,
        beta: float = 0.999,
        #**hparams,
    ):
        super().__init__()
        self.augment = default_augmentation(image_size) if augment_fn is None else augment_fn
        self.beta = beta
        self.encoder = EncoderWrapper(
            model, projection_size, hidden_size, layer=hidden_layer
        )
        self.predictor = nn.Linear(projection_size, projection_size, hidden_size)
        #self.hparams = hparams   #error is AttributeError: can't set attribute from call: byol = BYOL(model, image_size=(28, 28))  
        self._target = None

        self.encoder(torch.zeros(2, 3, *image_size))

    def forward(self, x: Tensor) -> Tensor:
        return self.predictor(self.encoder(x))

    @property
    def target(self):
        if self._target is None:
            self._target = deepcopy(self.encoder)
        return self._target

    def update_target(self):
        for p, pt in zip(self.encoder.parameters(), self.target.parameters()):
            pt.data = self.beta * pt.data + (1 - self.beta) * p.data

    # --- Methods required for PyTorch Lightning only! ---

    def configure_optimizers(self):
        optimizer = getattr(optim, self.hparams.get("optimizer", "Adam"))
        lr = self.hparams.get("lr", 1e-4)
        weight_decay = self.hparams.get("weight_decay", 1e-6)
        return optimizer(self.parameters(), lr=lr, weight_decay=weight_decay)

    def training_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]:
        x = batch[0]
        with torch.no_grad():
            x1, x2 = self.augment(x), self.augment(x)

        pred1, pred2 = self.forward(x1), self.forward(x2)
        with torch.no_grad():
            targ1, targ2 = self.target(x1), self.target(x2)
        loss = torch.mean(normalized_mse(pred1, targ2) + normalized_mse(pred2, targ1))

        self.log("train_loss", loss.item())
        self.update_target()

        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]:
        x = batch[0]
        x1, x2 = self.augment(x), self.augment(x)
        pred1, pred2 = self.forward(x1), self.forward(x2)
        targ1, targ2 = self.target(x1), self.target(x2)
        loss = torch.mean(normalized_mse(pred1, targ2) + normalized_mse(pred2, targ1))

        return {"loss": loss}

    ''' deprecated!!!!
    @torch.no_grad()
    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        val_loss = sum(x["loss"] for x in outputs) / len(outputs)
        self.log("val_loss", val_loss.item())
    '''

    '''
    @torch.no_grad()
    def on_validation_epoch_end(self, *arg, **kwargs) -> Dict:
        #val_loss = sum(x["loss"] for x in outputs) / len(outputs)
        all_preds = torch.stack(self.validation_step_outputs)
        #self.log("val_loss", val_loss.item())
    '''







model = resnet18(weights=True)
model.eval()
byol = BYOL(model, image_size=(28, 28))

trainer = pl.Trainer(
    max_epochs=50, 
    #gpus=-1,
    accumulate_grad_batches=2048 //32,  # // 128
    #weights_summary=None,
)
train_loader = DataLoader(
    TRAIN_DATASET,
    batch_size=32, #128
    shuffle=True,
    drop_last=True,
)
trainer.fit(byol, train_loader, val_loader)        

 
torch.save(model.state_dict(), 'byol2_pretrained_breastmnist_50_epochs.pth')  