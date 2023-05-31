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

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score


logger = TensorBoardLogger("logs/", name="tb_logger_sl_ft_pneumonia")


class SupervisedLightningModule(pl.LightningModule):
    def __init__(self, model: nn.Module, num_classes: int, **hparams):  
        super().__init__()
        self.model = model
        self.num_classes  = num_classes     

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def configure_optimizers(self):
        optimizer = getattr(optim, self.hparams.get("optimizer", "Adam"))
        lr = self.hparams.get("lr", 1e-4)
        weight_decay = self.hparams.get("weight_decay", 1e-6)
        return optimizer(self.parameters(), lr=lr, weight_decay=weight_decay)

    def training_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]:
        x, y = batch

        #few shot 
        support_x, support_y = next(iter(support_loader))
        x = torch.cat([x, support_x], dim=0)
        y = torch.cat([y, support_y], dim=0)

        y = y.float()
        loss = f.binary_cross_entropy_with_logits(self.forward(x), y)   
        self.log("train_loss", loss) #for tensorboard
        self.log("train_loss", loss.item())
        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]: 
        x, y = batch
        y = y.float()
        logits = self.forward(x)
        loss = f.binary_cross_entropy_with_logits(logits, y)

        # Calculate accuracy
        y_pred = (logits > 0).float()  # Convert logits to predicted labels
        accuracy = accuracy_score(y, y_pred)
        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy)

        # Calc AUC
        y_prob = torch.sigmoid(logits)
        auc = roc_auc_score(y.cpu().numpy(), y_prob.cpu().numpy())
        self.log("val_auc", auc)

        # Calculate precision, recall, and F1 score
        precision = precision_score(y.cpu().numpy(), y_pred.cpu().numpy())
        recall = recall_score(y.cpu().numpy(), y_pred.cpu().numpy())
        f1 = f1_score(y.cpu().numpy(), y_pred.cpu().numpy())

        self.log("test_precision", precision)
        self.log("test_recall", recall)
        self.log("test_f1", f1)


        return {"loss": loss}

    @torch.no_grad()
    def test_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]:
        x, y = batch
        y = y.float()
        logits = self.forward(x)
        loss = f.binary_cross_entropy_with_logits(logits, y)

        # Calculate accuracy
        y_pred = (logits > 0).float()  # Convert logits to predicted labels
        accuracy = accuracy_score(y, y_pred)
        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)

        # Calc AUC
        y_prob = torch.sigmoid(logits)
        auc = roc_auc_score(y.cpu().numpy(), y_prob.cpu().numpy())
        self.log("val_auc", auc)

        # Calculate precision, recall, and F1 score
        precision = precision_score(y.cpu().numpy(), y_pred.cpu().numpy())
        recall = recall_score(y.cpu().numpy(), y_pred.cpu().numpy())
        f1 = f1_score(y.cpu().numpy(), y_pred.cpu().numpy())

        self.log("test_precision", precision)
        self.log("test_recall", recall)
        self.log("test_f1", f1)

        return {"loss": loss}        


############### DATASETS - PNEUMONIA MNIST

print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

######### MY CONSTS
BATCH_SIZE = 32   
IMAGE_SIZE = 28 
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
NUM_WORKERS = multiprocessing.cpu_count()
print("NUM_WORKERS: ", NUM_WORKERS)

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
train_loader = DataLoader(dataset=TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(dataset=VAL_DATASET, batch_size=BATCH_SIZE, shuffle=False) 
test_loader = DataLoader(dataset=TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False) 


################## Supervised Training 
from torch.utils.data import DataLoader
from torchvision.models import resnet18
#Load stored model
model_path = '/home/ubuntu/remote-work/byol2/byol2_pretrained_chestmnist_batchsize_256_1000_epochs.pth'
print("Loading model: " + model_path)
saved_state_dict = torch.load(model_path)      
# Load the state dictionary into the model
model = resnet18()
model.load_state_dict(saved_state_dict)     

#I can experiment here and freeze 50%, 75%, 95% of layers....
# Do this if you want to update only the reshaped layer params, otherwise you will finetune the all the layers
#for param in model.parameters():
#    param.requires_grad = False
print("fine tuning all layers...")


num_features = model.fc.in_features
print("num_features: ", num_features)   #512
model.fc = nn.Linear(num_features, 1)   # output size of 1 is correct for binary classification


supervised = SupervisedLightningModule(model, num_classes=1) 
trainer = pl.Trainer(
    max_epochs=40, 
    logger=logger,
)


########### FEW SHOT ######## 
  
# Support set creation
support_samples_per_class = 5
support_indices = []
for class_label in range(n_classes):
    class_indices = [i for i, (_, label) in enumerate(TRAIN_DATASET) if label == class_label]
    selected_indices = random.sample(class_indices, support_samples_per_class)
    support_indices.extend(selected_indices)

support_dataset = torch.utils.data.Subset(TRAIN_DATASET, support_indices)
support_loader = DataLoader(
    dataset=support_dataset,
    batch_size=support_samples_per_class,
    shuffle=True,
    drop_last=True,
)


##############################




trainer.fit(supervised, support_loader, val_loader)

trainer.test(supervised, test_loader)












