import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import torch 


images = np.load('vae/1659672883257042/validation/reconstruction_epoch_34_env_0.npy')
input = images[0]
output = images[1]

fig, ax = plt.subplots(nrows = 8, ncols = 25)
fig.subplots_adjust(left  = 0.125,right = 0.9,bottom = 0.25,top = 0.75,wspace = 0.1,hspace = 0.1)
for k, i in enumerate(ax):
    for j, axis in enumerate(i):
        if (k%2==0):
            axis.axis('off')
            axis.imshow(input[int(k/2*25)+j].reshape(64,64,3))
            axis.set_xticklabels([])
            axis.set_yticklabels([])
            axis.set_aspect(1)
        if (k%2==1):
            axis.axis('off')
            axis.imshow(output[int((k-1)/2*25)+j].reshape(64,64,3))
            axis.set_xticklabels([])
            axis.set_yticklabels([])
            axis.set_aspect(1)
plt.savefig('recon_constant1.png')