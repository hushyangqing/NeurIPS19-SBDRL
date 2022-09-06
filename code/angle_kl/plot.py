import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def cal_weight(angle, up, cuda=True):
	
	tensor_0 = torch.zeros(1)
	tensor_1 = torch.ones(1)

	if up:
		return  torch.stack([
			torch.stack([torch.cos(angle), -torch.sin(angle), tensor_0, tensor_0]),
			torch.stack([torch.sin(angle), torch.cos(angle), tensor_0, tensor_0]),
			torch.stack([tensor_0, tensor_0, tensor_1, tensor_0]),
			torch.stack([tensor_0, tensor_0, tensor_0, tensor_1])
		]).reshape(4, 4)
	else: 
		return  torch.stack([
			torch.stack([tensor_1, tensor_0, tensor_0, tensor_0]),
			torch.stack([tensor_0, tensor_1, tensor_0, tensor_0]),
			torch.stack([tensor_0, tensor_0, torch.cos(angle), -torch.sin(angle)]),
			torch.stack([tensor_0, tensor_0, torch.sin(angle), torch.cos(angle)])
		]).reshape(4, 4)

vae = torch.load('vae/16615758883745522/saved_models/epoch_34_env_0', map_location={'cuda:0': 'cpu'})

print(vae.A_1_angle)
print(vae.A_2_angle)
print(vae.A_3_angle)
print(vae.A_4_angle)
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
