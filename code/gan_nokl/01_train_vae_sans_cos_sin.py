from constants import *
from torch.utils.data import DataLoader, Dataset
from torch.distributions import Normal
from vae.arch_torch_sans_cos_sin import VAE, CustomDataset
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os
import time
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import argparse
import os
from config import plot_figure_loss, early_stopping
from PIL import Image
from torch.autograd import Variable

img_shape = (3, 64, 64)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

def parse_args():
	desc = "Train VAE on one given maze environment"
	parser = argparse.ArgumentParser(description=desc)
	return parser.parse_args()

def set_relevant_grad_to_zero(A, up):

	if up:
		A.weight.grad[0][2:] = torch.Tensor([0.,0.]).cuda()
		A.weight.grad[1][2:] = torch.Tensor([0.,0.]).cuda()
		A.weight.grad[2] = torch.Tensor([0.,0.,0.,0.]).cuda()
		A.weight.grad[3] = torch.Tensor([0.,0.,0.,0.]).cuda()
	else:
		A.weight.grad[0] = torch.Tensor([0.,0.,0.,0.]).cuda()
		A.weight.grad[1] = torch.Tensor([0.,0.,0.,0.]).cuda()
		A.weight.grad[2][:2] = torch.Tensor([0.,0.]).cuda()
		A.weight.grad[3][:2] = torch.Tensor([0.,0.]).cuda()
		
	
	A.bias.grad = torch.Tensor([0.,0.,0.,0.]).cuda()
	
	return True


def loss_fn(recon_x, x, anneal):

	batch_size = x.size()[0]
	loss = F.mse_loss(recon_x, x)
	loss = torch.mean(loss)

	#print(F.mse_loss(torch.zeros(x.shape).float().cuda(), x), 'Loss with all black')

	# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	#kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

	return LAMBDA_RECONSTRUCTION * loss

def loss_fn_predict_next_z(pred, real):

	loss = F.mse_loss(pred, real)
	loss = torch.mean(loss)

	return loss

def train_batch(vae, optimizer, frames, actions, anneal, cuda=True):
	""" Train the VAE over a batch of example frames """

	optimizer.zero_grad()

	adversarial_loss = torch.nn.BCELoss()
	discriminator = Discriminator()

	if cuda:
		adversarial_loss.cuda()
		discriminator.cuda()

	Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
	valid = Variable(Tensor(frames.size(0), 1).fill_(1.0), requires_grad=False)
	fake = Variable(Tensor(frames.size(0), 1).fill_(0.0), requires_grad=False)

	real_imgs = Variable(frames.type(Tensor))

	recon_x, gen_imgs, z, z_plus_1 = vae(frames, actions)

	real_loss = adversarial_loss(discriminator(real_imgs), valid)
	fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
	d_loss = (real_loss + fake_loss)/2

	loss_recon = loss_fn(recon_x, frames, anneal)

	targets = z[1:]

	loss_predict_next_z = loss_fn_predict_next_z(pred=z_plus_1[:-1], real=targets)

	loss_total = loss_recon + loss_predict_next_z + d_loss

	loss_total.backward()

	set_relevant_grad_to_zero(A=vae.A_1, up=True)
	set_relevant_grad_to_zero(A=vae.A_2, up=True)
	set_relevant_grad_to_zero(A=vae.A_3, up=False)
	set_relevant_grad_to_zero(A=vae.A_4, up=False)
	
	optimizer.step()

	return float(loss_total), float(loss_recon), float(loss_predict_next_z), float(d_loss)

def train_batch_step(vae, discriminator, optimizer_D, optimizer_G, frames, actions, anneal, cuda=True):
	""" Train the VAE over a batch of example frames """

	adversarial_loss = torch.nn.BCELoss()

	if cuda:
		adversarial_loss.cuda()

	Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
	valid = Variable(Tensor(frames.size(0), 1).fill_(1.0), requires_grad=False)
	fake = Variable(Tensor(frames.size(0), 1).fill_(0.0), requires_grad=False)

	real_imgs = Variable(frames.type(Tensor))
	optimizer_G.zero_grad()

	recon_x, gen_imgs, z, z_plus_1 = vae(frames, actions)

	loss_recon = loss_fn(recon_x, frames, anneal)

	targets = z[1:]


	loss_predict_next_z = loss_fn_predict_next_z(pred=z_plus_1[:-1], real=targets)

	g_loss = loss_recon + loss_predict_next_z
	g_loss.backward()

	set_relevant_grad_to_zero(A=vae.A_1, up=True)
	set_relevant_grad_to_zero(A=vae.A_2, up=True)
	set_relevant_grad_to_zero(A=vae.A_3, up=False)
	set_relevant_grad_to_zero(A=vae.A_4, up=False)

	optimizer_G.step()

	optimizer_D.zero_grad()
	real_loss = adversarial_loss(discriminator(real_imgs), valid)
	fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
	d_loss = (real_loss + fake_loss)/2
	d_loss.backward()
	optimizer_D.step()

	loss_total = g_loss + d_loss

	return float(loss_total), float(loss_recon), float(loss_predict_next_z), float(d_loss)


def train_vae(folder):
	"""
	Train the VAE.
	"""

	total_ite = 0
	anneal = 1
	env = 0

	## Load or create models
	vae = VAE().to(DEVICE)
	discriminator = Discriminator().to(DEVICE)

	# Create optimizer
	# optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
	# optimizer for generator and discriminator
	optimizer_G = torch.optim.Adam(vae.parameters(), lr=1e-3, betas=(0.5, 0.999))
	optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-3, betas=(0.5, 0.999))
	## Load the dataset
	custom_dataset = CustomDataset(path_input='inputs.npy', path_action='actions.npy')
	# Define data loader
	dataset_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=BATCH_SIZE_VAE)

	# Tensorboard for monitoring
	writer = SummaryWriter(folder)
	fig_loss = []
	fig_loss_reco = []
	fig_loss_kl = []
	running_losses_mean = []
	early_stop = False

	for epoch in range(NUMBER_OF_EPOCHS_VAE):
		running_loss = []
		running_losses = []

		for batch_idx, (inputs, actions) in tqdm(enumerate(dataset_loader)):

			if batch_idx>2:
				pass

			##############################

			''' TRAINING '''

			inputs = inputs.float().cuda()
			actions = actions.long().cuda()

			# train and collect loss
			if epoch<0:
				anneal=1
			else:
				anneal = anneal*ANNEAL

			loss, loss_recon, loss_predict_next_z, d_loss= train_batch_step(vae, discriminator, optimizer_D, optimizer_G, inputs, actions, anneal)
			#import pdb 
			#pdb.set_trace()
			running_loss.append(loss)
			running_losses.append([loss, loss_recon, loss_predict_next_z, d_loss])

			##############################

			''' MONITORING & FIGURES '''

			# Monitoring for tensorboard

			if batch_idx % LOG_PERIOD == 0:

				# Log training loss.
				writer.add_scalar('loss/batch_loss', loss, batch_idx) 
				writer.add_scalar('loss/batch_loss_recon', loss_recon, batch_idx) 
				writer.add_scalar('loss/batch_loss_predict_next_z', loss_predict_next_z, batch_idx)

			# Figures

			if (batch_idx+1)% FIG_EARLY_STOP_PERIOD ==0:
				print('Anneal value:', anneal)
				running_losses_mean.append(np.mean(running_losses, axis=0))
				print('Loss: ',np.mean(running_losses, axis=0))
				#filename = folder + 'validation/losses_'+str(epoch)+'_env_'+str(env)+'.png'
				filename_n = folder + 'validation/normalized_losses_'+str(epoch)+'_env_'+str(env)+'.png'
				#plot_figure_loss(running_losses_mean, filename)
				#plot_figure_loss(running_losses_mean/np.max(running_losses_mean, axis=0), filename_n)

			##############################


			##############################

			''' EARLY STOPPING '''

			if (batch_idx+1)% FIG_EARLY_STOP_PERIOD ==0:

				early_stop = early_stopping(running_losses_mean=running_losses_mean, threshold=THRESHOLD_EARLY_STOPPING)

				if early_stop:
					pass

			##############################


		##############################

		''' MONITORING '''

		if epoch % 1 == 0:

			# Log training loss.
			writer.add_scalar('loss/epoch_loss', np.mean(running_loss), epoch) 

		# Monitor
		print('-------- Epoch', epoch, '/', NUMBER_OF_EPOCHS_VAE, 'finished. Loss on epoch : ', np.mean(running_losses, axis=0))

		##############################

		''' SAVING '''

		if epoch % SAVING_PERIOD == 0:

			torch.save(vae, folder+'saved_models/epoch_'+str(epoch)+'_env_'+str(env))

		##############################

		''' VALIDATING '''

		if epoch % VALIDATE_PERIOD == 0:


			obs_data = np.load('inputs.npy')[:100].reshape(-1,64,64,3).transpose((0,3,1,2))
			filename = folder + 'validation/reconstruction_epoch_'+str(epoch)+'_env_'+str(env)

			actions = np.load('actions.npy')[:100]
			actions = np.array(actions).reshape(-1,1)

			vae.generate_reconstructed_data(obs_data=obs_data, actions=actions, filename=filename)

		##############################

		if early_stop:
			pass


	writer.close() # close tensorboard



def main():

	# parse arguments
	args = parse_args()

	if args is None:
		print('Please give arguments!')
	
	folder = 'vae/' + str(time.time()).replace('.', '') + '/' # different folder for any experiment

	if not os.path.exists(folder):
		os.makedirs(folder)
		os.makedirs(folder+'saved_models')
		os.makedirs(folder+'validation')
		print('-------- Vizualize training with: tensorboard --port='+str(8006)+' --logdir='+folder+ ' runs')

	train_vae(folder) # Launch training


if __name__ == "__main__":
	main()
