import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

vae = torch.load('vae/1659492429474762/saved_models/epoch_34_env_0', map_location={'cuda:0': 'cpu'})
A_1 = vae.A_1.weight.detach().numpy()
A_2 = vae.A_2.weight.detach().numpy()
A_3 = vae.A_3.weight.detach().numpy()
A_4 = vae.A_4.weight.detach().numpy()
print(A_1)
print(np.linalg.det(A_1))
print(A_2)
print(np.linalg.det(A_2))
print(A_3)
print(np.linalg.det(A_3))
print(A_4)
print(np.linalg.det(A_4))

z = torch.tensor([1.,1.,1.,1.])
im = vae.forward(z,action=None,decode=True).detach().numpy().reshape(-1,3,64,64).transpose((0,2,3,1))
ims=[]
for ac in [-1,0,1,2]:
    aux=[]
    for i,action in enumerate(torch.Tensor(np.ones(15))+ac):
        im = vae.forward(z.cpu(),action=None, decode=True).detach().numpy().reshape(-1,3,64,64).transpose((0,2,3,1))
        im = im.reshape(64,64,3) 
        aux.append(im)
        action = action.long()
        next_z = vae.predict_next_z(z.reshape(-1,4),action.reshape(-1,1), cuda=False)
        z = next_z
    ims.append(aux)
    
fig, ax = plt.subplots(nrows=4, ncols=15)
fig.subplots_adjust(left  = 0.125,right = 0.9,bottom = 0.25,top = 0.75,wspace = 0.1,hspace = 0.1)
for k,i in enumerate(ax):
    for j,axis in enumerate(i):
        axis.axis('off')
        axis.imshow(ims[k][j])
        axis.set_xticklabels([])
        axis.set_yticklabels([])
        axis.set_aspect(1)
plt.savefig('constant1.png')
