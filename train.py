import os
import argparse
import multiprocessing
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

from byol_pytorch import BYOL
import pytorch_lightning as pl 

# resnet18
resnet = models.resnet18(pretrained=True)

# arguments

parser = argparse.ArgumentParser(description='byol-lightning-test')

parser.add_argument('--image_folder', type=str, required = True,
                       help='path to your folder of images for self-supervised learning')

args = parser.parse_args()

# constants

BATCH_SIZE = 32
EPOCHS     = 1000
LR         = 3e-4
#NUM_GPUS   = 2
IMAGE_SIZE = 28 #was 256
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
NUM_WORKERS = multiprocessing.cpu_count()

# pytorch lightning module

class SelfSupervisedLearner(pl.LightningModule):
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

    

# images dataset

def expand_greyscale(t):
    return t.expand(3, -1, -1)


class ImagesDataset(Dataset):
    def __init__(self, npz_file, image_size, set_flag):
        super().__init__()
        '''
        data = np.load('dermamnist.npz', allow_pickle=True)
        print(data.files)
        ->  ['train_images', 'val_images', 'test_images', 'train_labels', 'val_labels', 'test_labels']
        '''
        self.npz_file = npz_file
        self.image_size = image_size
        self.data = np.load(npz_file)

        if set_flag == 'train':
            self.images = self.data['train_images']    
            self.labels = self.data['train_labels']
        elif set_flag == 'validate':
            self.images = self.data['val_images']    
            self.labels = self.data['val_labels']
        elif set_flag == 'test':
            self.images = self.data['test_images']    
            self.labels = self.data['test_labels']

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale)
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        img = Image.fromarray(image)
        img = img.convert('RGB')
        return self.transform(img)


# main
if __name__ == '__main__':

    npz_file_path = '/home/ubuntu/.medmnist/chestmnist.npz' 

    ds_train = ImagesDataset(npz_file_path, IMAGE_SIZE, 'train')
    ds_validate = ImagesDataset(npz_file_path, IMAGE_SIZE, 'validate')
    ds_test = ImagesDataset(npz_file_path, IMAGE_SIZE, 'test')

    #ds = ImagesDataset(args.image_folder, IMAGE_SIZE)

    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    val_loader = DataLoader(ds_validate, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)


    model = SelfSupervisedLearner(
        resnet,
        image_size = IMAGE_SIZE,
        hidden_layer = 'avgpool',
        projection_size = 256,       # it is good to experiment with this size, tho it should be smaller than my input dimension of 28x28x3=2352
        projection_hidden_size = 4096,
        moving_average_decay = 0.99
    )

    trainer = pl.Trainer(
        #gpus = NUM_GPUS,
        max_epochs = EPOCHS,
        accumulate_grad_batches = 1,
        sync_batchnorm = True  #should be true for distributed training(as per momentum^2 paper
    )


    # Set float32 matrix multiplication precision
    torch.set_float32_matmul_precision('high') 

    #I may have misteknly only used the training set (80K instead of 112K)from ChestMNIST instead of all images, prob ok in case i dont want to run the 1000 epochs again
    trainer.fit(model, train_loader)

    torch.save(model.state_dict(), 'byol_pretrained_chestmnist_1000_epochs.pth')   #was model.

    #you dont validate pre-training silly, but lets viz with t 
    #trainer.validate(model, dataloaders=val_loader)
    #trainer.test(model, test_loader)
