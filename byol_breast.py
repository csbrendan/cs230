import torch
from byol import BYOL
from torchvision import models
import numpy as np

#architecture
resnet = models.resnet18(pretrained=True)

learner = BYOL(
    resnet,
    image_size = 28,
    hidden_layer = 'avgpool'
)

#optimizer
opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

def sample_unlabelled_images():
    #/home/ubuntu/.medmnist/breastmnist.npz
    npz_file = '/home/ubuntu/.medmnist/breastmnist.npz'
    data = np.load(npz_file)
    images = data['train_images'] 
    #labels = data['train_labels'] #where we're going we dont need .... labels
    

    return torch.randn(20, 3, 28, 28)

for _ in range(100):
    images = sample_unlabelled_images()
    loss = learner(images)
    opt.zero_grad()
    loss.backward()
    opt.step()
    learner.update_moving_average() # update moving average of target encoder

# save your improved network
torch.save(resnet.state_dict(), './improved-net.pt')