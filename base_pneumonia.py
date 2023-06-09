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

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

from scipy.stats import sem
from scipy.stats import norm

logger = TensorBoardLogger("logs/", name="tb_logger_sl_ft_pneumonia")


class SupervisedLightningModule(pl.LightningModule):
    def __init__(self, model: nn.Module, num_classes: int, **hparams): 
        super().__init__()
        self.model = model
        self.num_classes  = num_classes   
        self.true_labels = []
        self.predicted_labels = []

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def configure_optimizers(self):
        optimizer = getattr(optim, self.hparams.get("optimizer", "Adam"))
        lr = self.hparams.get("lr", 0.001)    #for base models
        weight_decay = self.hparams.get("weight_decay", 1e-6) #L2 regularization 1e-6   (log scale 1e-6 -> 1e-3)
        return optimizer(self.parameters(), lr=lr, weight_decay=weight_decay)

    def training_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]:
        x, y = batch
        y = y.float()
        loss = f.binary_cross_entropy_with_logits(self.forward(x), y)   
        self.log("train_loss", loss) 
        self.log("train_loss", loss.item())
        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]: 
        x, y = batch
        y = y.float()
        logits = self.forward(x)
        loss = f.binary_cross_entropy_with_logits(logits, y) 

        y_pred = (logits > 0).float()  # Convert logits to predicted labels

        # accuracy
        accuracy = accuracy_score(y.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy)

        # AUC
        y_prob = torch.sigmoid(logits)
        auc = roc_auc_score(y.cpu().detach().numpy(), y_prob.cpu().detach().numpy(), average='macro')
        self.log("val_auc", auc)

        # precision, recall, and F1 score
        precision = precision_score(y.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), average='macro')
        recall = recall_score(y.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), average='macro')
        f1 = f1_score(y.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), average='macro')

        self.log("val_precision", precision)
        self.log("val_recall", recall)
        self.log("val_f1", f1)

        return {"loss": loss}

    @torch.no_grad()
    def test_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]:
        x, y = batch
        y = y.float()
        logits = self.forward(x)
        loss = f.binary_cross_entropy_with_logits(logits, y)
        self.log("test_loss", loss)

        # default decision
        y_pred = (logits > 0).float()  # Convert logits to predicted labels

        # accumulate true labels and predicted labels for confusion matrix
        self.true_labels.append(y.cpu().detach().numpy())
        self.predicted_labels.append(y_pred.cpu().detach().numpy())

        return {"loss": loss}


    def on_test_end(self) -> None:
        # compute confusion matrix using accumulated labels
        true_labels = np.concatenate(self.true_labels) #y
        predicted_labels = np.concatenate(self.predicted_labels) #y_pred
        confusion = confusion_matrix(true_labels, predicted_labels)
        print("Confusion Matrix")
        print(confusion)

        # accuracy
        accuracy = accuracy_score(true_labels, predicted_labels)

        # AUC
        auc = roc_auc_score(true_labels, predicted_labels, average='macro')
        
        # confidence interval using the non-parametric method by DeLong
        n = len(true_labels)
        auc_var = auc * (1 - auc)
        auc_se = np.sqrt(auc_var / n)
        # calc lower and upper bounds of the confidence interval
        alpha = 0.95  # desired confidence level
        z = norm.ppf(1 - (1 - alpha) / 2)
        lower_bound = auc - z * auc_se
        upper_bound = auc + z * auc_se

        # precision, recall, and F1 score
        precision = precision_score(true_labels, predicted_labels, average='macro')
        recall = recall_score(true_labels, predicted_labels, average='macro')
        f1 = f1_score(true_labels, predicted_labels, average='macro')

        print("accuracy: ", accuracy)
        print("precision: ", precision)
        print("recall: ", recall)
        print("f1: ", f1)
        print("auc: ", auc)

        print("CI - lower bound: ", lower_bound)
        print("CI - upper bound: ", upper_bound)


############### DATASETS - PNEUMONIA MNIST

print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

# CONSTS
BATCH_SIZE = 32   
IMAGE_SIZE = 28 
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
NUM_WORKERS = multiprocessing.cpu_count()
print("NUM_WORKERS: ", NUM_WORKERS)

from torchvision.transforms import ToTensor

data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert to RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
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
val_loader = DataLoader(dataset=VAL_DATASET, batch_size=BATCH_SIZE, shuffle=False) 
test_loader = DataLoader(dataset=TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False) 




################## Supervised Training again
from torch.utils.data import DataLoader
from torchvision.models import resnet18
     
# Load the state dictionary into the model
print("load basemodel pretrained Resnet-18")
model = resnet18(weights=True)

# experiment here and freeze 50%, 75%, 95% of layers....
# do this if you want to update only the reshaped layer params, otherwise you will finetune the all the layers
#for param in model.parameters():
#    param.requires_grad = False
print("fine tuning all layers...")


num_features = model.fc.in_features
print("num_features: ", num_features)   
model.fc = nn.Linear(num_features, 1)   


supervised = SupervisedLightningModule(model, num_classes=1) 
trainer = pl.Trainer(
    max_epochs=25,
    logger=logger,
)
train_loader = DataLoader(
    TRAIN_DATASET,
    batch_size=32,
    shuffle=True,
    drop_last=True,
)
trainer.fit(supervised, train_loader, val_loader)

trainer.test(supervised, test_loader)

