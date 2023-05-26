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

#Mine
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

import medmnist
from medmnist import INFO, Evaluator

import os
import argparse
import multiprocessing
from pathlib import Path
from PIL import Image
import numpy as np



class SupervisedLightningModule(pl.LightningModule):
    def __init__(self, model: nn.Module, num_classes: int, **hparams):  #added num_classes since STL10 is 1000 classes
        super().__init__()
        self.model = model
        self.num_classes  = num_classes     #me, since not STL10

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def configure_optimizers(self):
        optimizer = getattr(optim, self.hparams.get("optimizer", "Adam"))
        lr = self.hparams.get("lr", 1e-4)
        weight_decay = self.hparams.get("weight_decay", 1e-6)
        return optimizer(self.parameters(), lr=lr, weight_decay=weight_decay)

    def training_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]:
        x, y = batch
        #y = y.unsqueeze(1)
        #y = f.one_hot(y, num_classes=self.num_classes).float()
        y = y.float()
        loss = f.binary_cross_entropy_with_logits(self.forward(x), y)   #loss = f.cross_entropy(self.forward(x), y)
        self.log("train_loss", loss.item())
        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]: 
        x, y = batch

        #print("inputs is ")
        #print(x)  # [[[[[ 0.0902,  0.0902,  0.0824,  ...,  0.2941,  0.2706,  0.2941],   INPUT is normalized images...
        print("targets is ")
        print(y)  # [ [1],[1],[1],[0], .... TARGET is labels 

        print("inputs shape is ")
        print(x.shape) # [64, 3, 28, 28]     2 batches of size 32..(64)  of 28x28x3 images       
        print("targets shape is ")
        print(y.shape) # [64, 1]             2 batches of size 32 of labels 

        '''
        y = y.unsqueeze(1)
        y = f.one_hot(y, num_classes=self.num_classes).float()             #me since not STL10
        print("unsqueezed & 1-hot targets is ")
        print(y)  # [ [1],[1],[1],[0], .... labels 
        print("unsqueezed & 1-hot targets shape is ")
        print(y.shape) # [64, 1]             2 batches of size 32 of labels 
        '''
        y = y.float()
        loss = f.binary_cross_entropy_with_logits(self.forward(x), y) #was cross_entropy()
        return {"loss": loss}


############### DATASETS - PNEUMONIA MNIST

print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

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

data_flag = 'pneumoniamnist'
info = INFO[data_flag]
print("medMNIST dataset INFO: ")
print(info)

task = info['task']
n_channels = info['n_channels']
print("n_channels: ", n_channels)   # 1
n_classes = len(info['label'])
print("n_classes: ", n_classes)     # 2

DataClass = getattr(medmnist, info['python_class'])
TRAIN_DATASET = DataClass(split='train', transform=data_transform, download=False)
VAL_DATASET = DataClass(split='val', transform=data_transform, download=False)
TEST_DATASET = DataClass(split='test', transform=data_transform, download=False)



# encapsulate data into dataloader form
train_loader = DataLoader(dataset=TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=VAL_DATASET, batch_size=BATCH_SIZE, shuffle=False) # was 2*BATCH_SIZE
test_loader = DataLoader(dataset=TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False) # was 2*BATCH_SIZE




################ Supervised Training without BYOL

# dont need

#################################################



################## Supervised Training again
from torch.utils.data import DataLoader
from torchvision.models import resnet18
#Load stored model
model_path = '/home/ubuntu/remote-work/byol2/byol2_pretrained_breastmnist_50_epochs.pth'
saved_state_dict = torch.load(model_path)      
# Load the state dictionary into the model
model = resnet18()
model.load_state_dict(saved_state_dict)     #model.load_state_dict(torch.load(model_path, map_location=device)['net'], strict=True)

# Do this if you want to update only the reshaped layer params, otherwise you will finetune the all the layers
#for param in model.parameters():
#    param.requires_grad = False

num_features = model.fc.in_features
print("num_features: ", num_features)   #512
model.fc = nn.Linear(num_features, 1)   # output size of 1 is correct for binary classification

supervised = SupervisedLightningModule(model, num_classes=1) 
trainer = pl.Trainer(
    max_epochs=25, 
    #gpus=-1,
    #weights_summary=None,
)
train_loader = DataLoader(
    TRAIN_DATASET,
    batch_size=32,
    shuffle=True,
    drop_last=True,
)
trainer.fit(supervised, train_loader, val_loader)



def accuracy(pred: Tensor, labels: Tensor) -> float:
    return (pred.argmax(dim=-1) == labels).float().mean().item()


#model.cuda()
#acc = sum([accuracy(model(x.cuda()), y.cuda()) for x, y in val_loader]) / len(val_loader)

#model.cpu()
acc = sum([accuracy(model(x), y) for x, y in val_loader]) / len(val_loader)
print(f"Accuracy: {acc:.3f}")




   


def test(split):
    model.eval()  #was model.
    y_true = torch.tensor([])
    y_score = torch.tensor([])
    
    data_loader = val_loader if split == 'train' else test_loader

    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                outputs = outputs.softmax(dim=-1)
            else:  #should be for binary-class
                targets = targets.squeeze().long()
                outputs = outputs.softmax(dim=-1)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.numpy()
        y_score = y_score.detach().numpy()
        
        evaluator = Evaluator(data_flag, split)
        metrics = evaluator.evaluate(y_score)
    
        print('%s  auc: %.3f  acc:%.3f' % (split, *metrics))

        
print('==> Evaluating ...')
#test('train')
test('test') 







