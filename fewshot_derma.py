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
from torch.utils.data import Dataset, DataLoader, Subset

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
from sklearn.metrics import confusion_matrix

logger = TensorBoardLogger("logs/", name="tb_logger_sl_ft_derma")


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
        lr = self.hparams.get("lr", 1e-4)
        weight_decay = self.hparams.get("weight_decay", 1e-6)
        return optimizer(self.parameters(), lr=lr, weight_decay=weight_decay)

    def training_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]:
        x, y = batch

        #few shot 
        support_x, support_y = next(iter(support_loader))
        x = torch.cat([x, support_x], dim=0)
        y = torch.cat([y, support_y], dim=0)

        y = y.long()
        #loss = f.binary_cross_entropy_with_logits(self.forward(x), y)   
        loss = torch.nn.CrossEntropyLoss()(self.forward(x), torch.squeeze(y))         
        self.log("train_loss", loss) #for tensorboard
        self.log("train_loss", loss.item())
        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]: 
        x, y = batch
        y = y.long()
        logits = self.forward(x)
        loss = torch.nn.CrossEntropyLoss()(logits, torch.squeeze(y)) 

        # Calculate accuracy
        _, y_pred = torch.max(logits, dim=1)    # obtain predicted class labels
        accuracy = accuracy_score(y.cpu().numpy(), y_pred.cpu().numpy())
        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy)

        # Calculate precision, recall, and F1 score
        precision = precision_score(y.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
        recall = recall_score(y.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
        f1 = f1_score(y.cpu().numpy(), y_pred.cpu().numpy(), average='macro') 
        self.log("val_precision", precision)
        self.log("val_recall", recall)
        self.log("val_f1", f1)

        return {"loss": loss}

    @torch.no_grad()
    def test_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]:
        x, y = batch
        y = y.long()
        logits = self.forward(x)
        loss = torch.nn.CrossEntropyLoss()(logits, torch.squeeze(y)) 
        
        # Calculate accuracy
        _, y_pred = torch.max(logits, dim=1) 
        accuracy = accuracy_score(y.cpu().numpy(), y_pred.cpu().numpy())
        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)

        # accumulate true labels and predicted labels for confusion matrix
        self.true_labels.append(y.cpu().numpy())
        self.predicted_labels.append(y_pred.cpu().numpy())        

        # Calculate precision, recall, and F1 score
        precision = precision_score(y.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
        recall = recall_score(y.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
        f1 = f1_score(y.cpu().numpy(), y_pred.cpu().numpy(), average='macro') 

        self.log("test_precision", precision)
        self.log("test_recall", recall)
        self.log("test_f1", f1)

        return {"loss": loss}     


    def on_test_end(self) -> None:
        # compute confusion matrix using accumulated labels
        true_labels = np.concatenate(self.true_labels)
        predicted_labels = np.concatenate(self.predicted_labels)
        confusion = confusion_matrix(true_labels, predicted_labels)
        print("Confusion Matrix")
        print(confusion)       


############### DATASETS - DERMA MNIST

print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

######### MY CONSTS
BATCH_SIZE = 32   
IMAGE_SIZE = 28 
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
NUM_WORKERS = multiprocessing.cpu_count()
print("NUM_WORKERS: ", NUM_WORKERS)

from torchvision.transforms import ToTensor

data_transform = transforms.Compose([
    #transforms.Grayscale(num_output_channels=3),  # Convert to RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize for RGB
])

data_flag = 'dermamnist'
info = INFO[data_flag]
print("medMNIST dataset INFO: ")
print(info)

task = info['task']
n_channels = info['n_channels']
print("n_channels: ", n_channels)   # 1 ...... 3
n_classes = len(info['label'])
print("n_classes: ", n_classes)     # 2 ...... 7

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
model.fc = nn.Linear(num_features, 7)   # output size of 7 for multi class

supervised = SupervisedLightningModule(model, num_classes=7) 
trainer = pl.Trainer(
    max_epochs=25, 
    logger=logger,
)

########### FEW SHOT ######## 
data_3 = np.load('/home/ubuntu/.medmnist/dermamnist.npz', allow_pickle=True)
train_labels = data_3['train_labels']
train_labels = train_labels.flatten()

# get the unique label values
unique_labels = np.unique(train_labels)

# get num of samples in each class
class_counts_3 = np.bincount(train_labels)

# store the desired number of samples for each label
samples_per_label = 100

# iterate through the unique label values
support_indices = []
for label in unique_labels:

    '''
    label_indices = np.where(train_labels == label)[0]
    positive_indices = np.where(label_indices < class_counts_3[label])[0]
    num_positive_samples = len(positive_indices)
    num_samples_to_select = min(samples_per_label, num_positive_samples)
    selected_indices = np.random.choice(positive_indices, size=num_samples_to_select, replace=False)
    support_indices.extend(label_indices[selected_indices])
    '''  
        
# create a Subset of the original dataset using the selected indices
support_dataset = Subset(TRAIN_DATASET, support_indices)

# create the DataLoader
support_loader = DataLoader(
    dataset=support_dataset,
    batch_size=samples_per_label,
    shuffle=True,
    drop_last=True,
)
##############################

trainer.fit(supervised, support_loader, val_loader)

trainer.test(supervised, test_loader)













