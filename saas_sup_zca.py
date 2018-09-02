import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.benchmark = True # will turn on the cudnn autotuner that selects efficient algorithms.
nll_loss = nn.NLLLoss(size_average=True)
logsoft = nn.LogSoftmax() if torch.__version__[2]=="1" else nn.LogSoftmax(dim=1)

import os
import time
import numpy as np
import argparse
np.set_printoptions(precision=3)

from manual_load import train_w, train_yu, test
from saas_helper import setup

### get dataset and network name as argument
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='svhn' or 'cifar10')
parser.add_argument('--net_name', help='conv_large' or 'resnet')
args = parser.parse_args()
dataset = args.dataset; net_name = args.net_name
assert(dataset in ['cifar10'])
assert(net_name in ['conv_large'])

file_zca = "data/cifar10_whitened.npz"
if not os.path.exists(file_zca):      
    from zca import whiten_cifar10       
    whiten_cifar10(file_zca)

### hyperparameters:
lr = 0.1 # learning rate for weights 
batch_size = 100 # batch size for both labeled and unlabeled data
unlab_rat = 1.0 # Percentage of the unlabeled data to be used. If 1.0, entire training data is used. 
y_u_folder = 'estimated_labels/' # folder to save the y_u estimates
count = 1 # counter for number of epochs best validation did no increase
halve = 50 # halve the learning rate whenever count exceeds halve
num_epochs = 3500 # maximum possible number of epochs
wd = 10**-4 # weight decay
stop_lr = 10**-4 # stop training when learning rate less than or equal to stop_lr
is_whiten = True

### setup directories and set dataset/net_name dependent parameters:
net_w, augment_type, nb_labelled, _, _ = setup(dataset, net_name, [y_u_folder])
y_file = '%s_%s'%(dataset, net_name) # path to the .npy file from which y_u loaded. If "", start with random estimates.
file_zca = file_zca = "data/cifar10_whitened.npz"

### load y_u estimates and take the argmax:
state = np.load("estimated_labels/"+y_file+".npy").item()
lab_inds = state.get("lab_inds")
estimated_y = state.get("estimated_y")
yu = np.argmax(estimated_y, axis=1)

### load dataloaders:
if dataset == 'cifar10':   
    from zca import get_loaders_cifar_zca
    trainloader_l, testloader, trainloader_u, trainset_l, test_set, trainset_u, lab_inds = \
    get_loaders_cifar_zca(nb_labelled, batch_size, is_whiten, file_zca, lab_inds)    
    train_labels_u = trainset_u.train_labels
    train_data_u_np = trainset_u.train_data
else:
    raise NotImplementedError

assert(yu.shape == np.array(train_labels_u).shape) 
y_acc = ((yu==np.array(train_labels_u)).sum())*1.0/len(train_labels_u)    
print("est_y_arg acc=%f" % (y_acc))
yu = torch.LongTensor(yu).cuda()

best_acc = 0
for epoch in range(num_epochs):
    st_time = time.time() 
    
    ### update the learning rate: 
    if count > halve:
        lr /= 2.0; count = 1 
    optimizer_w = optim.SGD([{'params': net_w.parameters()}], lr, momentum=0.9, weight_decay=wd)

    ### train with labeled data:    
    acc_l, loss = train_w(net_w, optimizer_w, nll_loss, trainset_l.train_data, trainset_l.train_labels, batch_size)

    ### train with unlabeled data and SaaS estimates: 
    acc_u, loss_u = train_yu(net_w, optimizer_w, nll_loss, trainset_u.train_data, train_labels_u, yu, batch_size)    
    
    ### run test to report the validation accuracy:
    prev_best_acc = best_acc
    valid_acc, best_acc = test(net_w, test_set.test_data, test_set.test_labels, best_acc, batch_size)

    print(y_file)
    print("epoch=%d,time=%f,lr=%f,acc_l=%f,acc_u=%f,valid_acc=%f,best_acc=%f" %\
          (epoch, time.time()-st_time, lr, acc_l, acc_u, valid_acc, best_acc))
    
    ### update the count:
    count = 1 if prev_best_acc < best_acc else count + 1    
    if lr < stop_lr:
        break