from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import pytorch_lightning as pl 

import random
import medmnist
from medmnist import INFO, Evaluator

import os
import argparse
import multiprocessing
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from byol_pytorch import BYOL

print("TESTING *************")

#choose dataset and wether to download again
data_flag = 'pneumoniamnist'
#download = True
download = False


#constants from train.py
BATCH_SIZE = 32
EPOCHS     = 10 #was 1000
LR         = 3e-4
IMAGE_SIZE = 28 #was 256
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
#NUM_WORKERS = multiprocessing.cpu_count()
#####


#NUM_EPOCHS = 5
#BATCH_SIZE = 128
lr = 0.001

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

#Preprocessing
#because ResNet expects RGB not grayscale, we need to convert the images to (normalized) RGB
data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert to RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize for RGB
])

# load the data
train_dataset = DataClass(split='train', transform=data_transform, download=download)
test_dataset = DataClass(split='test', transform=data_transform, download=download)

# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

#print(train_dataset)
#print("===================")
#print(test_dataset)

######## TRANSFER

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
#data_dir = 

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "chestmnist"

# Number of classes in the dataset
num_classes = 2

# Batch size for training (change depending on how much memory you have)
#batch_size = 8

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class FineTuneModel(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)

    def forward(self, images):
        return self.learner(images)

    def training_step(self, images, _):
        loss = self.forward(images)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()



# CHESTMNIST_BYOL transfer learn to simple CNN model
#model.fc = nn.Linear(512, num_classes)
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0

    if model_name == "chestmnist":
        #model_path = '/home/ubuntu/remote-work/byol/byol_pretrained_chestmnist_100epochs_matmul_low.pth'
        #model_ft = torch.load(model_path)



        resnet = models.resnet18(pretrained=False)

        model_ft = FineTuneModel(
            resnet,
            image_size = IMAGE_SIZE,
            hidden_layer = 'avgpool',
            projection_size = 256,       # it is good to experiment with this size, tho it should be smaller than my input dimension of 28x28x3=2352
            projection_hidden_size = 4096,
            moving_average_decay = 0.99
        )

        # Load the saved model state dictionary
        model_path = '/home/ubuntu/remote-work/byol/byol_pretrained_chestmnist_100epochs_matmul_low.pth'
        saved_state_dict = torch.load(model_path)
        # Load the state dictionary into the model
        model_ft.load_state_dict(saved_state_dict)

        #set_parameter_requires_grad(model_ft, feature_extract)

        num_features = model_ft.learner.net.fc.in_features  #AttributeError("'{}' object has no attribute '{}'".format( AttributeError: 'FineTuneModel' object has no attribute 'fc'
        print("num_ftrs " + str(num_features))
        model_ft.learner.net.fc = nn.Linear(num_features, 2)
        input_size = 224


    return model_ft, input_size

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
print(model_ft)

# instantiate the model above
#resnet_output_size = model_ft.fc.in_features
# Replace the last layers of Resnet model_ft with the layers from net_model
#model_ft.fc = model_ft.fc

# define loss function and optimizer
task = 'binary-class'
if task == "binary-class":
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()
    
optimizer = optim.SGD(model_ft.parameters(), lr=lr, momentum=0.9)


# train

for epoch in range(EPOCHS):
    train_correct = 0
    train_total = 0
    test_correct = 0
    test_total = 0
    
    model_ft.train()
    for inputs, targets in tqdm(train_loader):
        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = model_ft(inputs) #was model()
        
        
        if task == 'binary-class':
            print("task is " + task)



            print("targets is " + str(targets))
            targets = targets.to(torch.float32)
            print("targets is now " + str(targets))

            print("outputs is " + str(outputs))

            targets = targets.squeeze(1)    #targets = targets.unsqueeze(1)
            outputs = outputs.unsqueeze(0) 
            print("len output shape")
            print(len(outputs.shape))


            print("outputs shape:", outputs.shape)
            print("targets shape:", targets.shape)
            print("**********")

            loss = criterion(outputs, targets)    # !!!


        loss.backward()
        optimizer.step()


#save model to disk
#torch.save(model_ft.state_dict(), 'byol_chestmnist_100epochs_pneumoniamnist.pth')   #was model.

#load saved model
#restored_model = torch.load(PATH)


def test(split):
    model_ft.eval()  #was model.
    y_true = torch.tensor([])
    y_score = torch.tensor([])
    
    data_loader = train_loader_at_eval if split == 'train' else test_loader

    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model_ft(inputs)

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                outputs = outputs.softmax(dim=-1)
            else:
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
#test('test')        