import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.benchmark = True # will turn on the cudnn autotuner that selects efficient algorithms.

import os
import time
import numpy as np
from helper import get_loaders_cifar, test, train_w
from resnet import ResNet18
	
nll_loss = nn.NLLLoss(size_average=True) 

def train_on_noisy(noise_rat=1.0, num_epochs=30, nb_labelled = 50000, batch_size = 100, \
	wd=0.0, lr = 0.1, dataset = 'cifar10'):

	net_w = ResNet18().cuda() 
	optimizer_w = optim.SGD([{'params': net_w.parameters()}], lr, momentum=0.9, weight_decay=wd)

	trainloader_l = get_loaders_cifar(nb_labelled, dataset, batch_size, noise_rat)

	# get the loss for the random weight to have equal starts:
	_, loss = test(net_w, nll_loss, trainloader_l)
	losses = [loss[0]]

	for epoch in range(num_epochs):
		st_time = time.time() 
		_, loss = train_w(net_w, optimizer_w, nll_loss, trainloader_l)
		losses.append(loss[0])
		print ("noise_rat=%f, epoch=%d, time=%f, loss=%f" % \
			(noise_rat, epoch, time.time()-st_time, loss[0]))
	return losses


file_name = "training_losses.npy"
for i in range(3): # number of trainings to average over per each noise ratio 
	for noise_rat in [0.0, 0.25, 0.5, 0.75, 1.0]:
		losses = train_on_noisy(noise_rat)

		# check if .npy file created:
		logs = np.load(file_name)[()] if os.path.exists(file_name) else {}
		  
		# update the logs
		key = str(noise_rat)
		if key in logs.keys():
			logs[key].append(losses)
		else:
			logs.update({key:[losses]})
		  
		# save to .npy file:
		logs = np.save(file_name, logs)


