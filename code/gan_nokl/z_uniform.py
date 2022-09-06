# show the distribution of images in the latent space

import torch
import os
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import random

from vae.arch_torch_sans_cos_sin import VAE, CustomDataset 
## CUDA variable from Torch
CUDA = torch.cuda.is_available()
#torch.backends.cudnn.deterministic = True
## Dtype of the tensors depending on CUDA
GPU_MODE = True
DEVICE = torch.device("cuda") if GPU_MODE else torch.device("cpu")
vae = torch.load('vae/16620802335176122/saved_models/epoch_34_env_0', map_location={'cuda:0': 'cpu'}).to(DEVICE)

# t is the temperature??? so
def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

custom_dataset = CustomDataset(path_input = 'inputs.npy', path_action = 'actions.npy')
dataset_loader = torch.utils.data.DataLoader(dataset=custom_dataset)

z_all = []
z_1 = []
z_2 = []
z_3 = []
z_4 = []
z_plus_all = []
z_plus_1 = []
z_plus_2 = []
z_plus_3 = []
z_plus_4 = []
for batch_indx, (inputs, actions) in enumerate(dataset_loader):
    input = inputs.float().cuda()
    action = actions.float().cuda()
    z_res = vae.forward(input, action, encode=True)
    z = z_res[0].reshape(4).tolist()
    z_plus = z_res[1].reshape(4).tolist()
    z_all.append(z)
    z_plus_all.append(z_plus)

z_final = random.sample(z_all, 1000)
z_1 = [k[0] for k in z_final]
z_2 = [k[1] for k in z_final]
z_3 = [k[2] for k in z_final]
z_4 = [k[3] for k in z_final]

ax = plt.axes(projection='3d')
ax.scatter3D(z_1, z_2, z_3, cmap='Blues', alpha=0.6)
plt.savefig('visual_constant1000_step.png')

ax1 = plt.axes(projection='3d')
ax1.scatter3D(z_1, z_2, z_4, cmap='Blues', alpha=0.6)
plt.savefig('visual4_constant1000_step.png')

