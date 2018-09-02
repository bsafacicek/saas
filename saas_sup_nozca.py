import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
cudnn.benchmark = True # will turn on the cudnn autotuner that selects efficient algorithms.
nll_loss = nn.NLLLoss(size_average=True) 
logsoft = nn.LogSoftmax() if torch.__version__[2]=="1" else nn.LogSoftmax(dim=1)

import time
import numpy as np
import copy
import argparse
np.set_printoptions(precision=3)
    
from saas_helper import train_w, test, setup

###  get dataset and network name as argument
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='svhn' or 'cifar10')
parser.add_argument('--net_name', help='conv_large' or 'resnet')
args = parser.parse_args()
dataset = args.dataset; net_name = args.net_name
assert(dataset in ['svhn', 'cifar10'])
assert(net_name in ['conv_large', 'resnet'])

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

### setup directories and set dataset/net_name dependent parameters:
net_w, augment_type, nb_labelled, _, _ = setup(dataset, net_name, [y_u_folder])
y_file = '%s_%s'%(dataset, net_name) # path to the .npy file from which y_u loaded. If "", start with random estimates.

### load y_u estimates and take the argmax:
state = np.load("estimated_labels/"+y_file+".npy").item()
lab_inds = state.get("lab_inds")
estimated_y = state.get("estimated_y")
yu = np.argmax(estimated_y, axis=1)

### load dataloaders:
if dataset == 'svhn':
    from datasets_svhn import get_loader_svhn, shuffle_loader_svhn
    trainloader_l, testloader, trainloader_u, trainset_l, test_set, trainset_u_org, lab_inds = \
    get_loader_svhn(nb_labelled, batch_size, unlab_rat, augment_type, lab_inds)
    train_labels_u_org = trainset_u_org.labels
    train_data_u_np = trainset_u_org.data
elif dataset == 'cifar10':   
    from datasets_cifar import get_loaders_cifar, shuffle_loader_cifar
    trainloader_l, testloader, trainloader_u, trainset_l, test_set, trainset_u_org, lab_inds = \
    get_loaders_cifar(nb_labelled, batch_size, unlab_rat, augment_type, lab_inds)
    train_labels_u_org = trainset_u_org.train_labels
    train_data_u_np = trainset_u_org.train_data
    
assert(yu.shape == np.array(train_labels_u_org).shape) 
y_acc = ((yu==np.array(train_labels_u_org)).sum())*1.0/len(train_labels_u_org)    
print("est_y_arg acc=%f" % (y_acc))
yu = torch.LongTensor(yu).cuda()
  
def train(net_w, optimizer_w, criterion, trainloader_u, est_ys):
    net_w.train()
    total_u, correct_u = 0, 0

    ### permute entire trainloader for unlabeled data and shuffling indexes to be used with estimated_y
    if dataset == 'svhn':        
        trainloader_u, ind_shuff_all = shuffle_loader_svhn(trainset_u_org, batch_size) 
    elif dataset == 'cifar10':   
        trainloader_u, ind_shuff_all = shuffle_loader_cifar(trainset_u_org, batch_size) 

    for batch_idx, (inps_u, targs_u_org) in enumerate(trainloader_u): # targs_u_org used only as post mortem
        ### get correct labels for post mortem:
        ind_shuff = copy.deepcopy(ind_shuff_all[batch_idx*batch_size:(batch_idx+1)*batch_size])     
        targs_u = est_ys[ind_shuff]
        
        ### ground truth for post mortem analysis:
        targs_u_org = copy.deepcopy(train_labels_u_org[ind_shuff.cpu().numpy()])         
        targs_u_org = torch.from_numpy(targs_u_org)    
        
        ### calculate objective funtion:
        inps_u, targs_u = Variable(inps_u.cuda()), Variable(targs_u.cuda())
        outs_u = logsoft(net_w(inps_u))
        loss = nll_loss(outs_u, targs_u)  
        loss.backward(); optimizer_w.step(); optimizer_w.zero_grad()
        
        ### acc_u:
        _, predicted_u = torch.max(outs_u.data, 1)
        total_u += targs_u_org.size(0)
        correct_u += predicted_u.cpu().eq(targs_u_org).sum()
    
    return 100.*correct_u/total_u

best_acc = 0
for epoch in range(num_epochs):
    st_time = time.time() 
    
    ### update the learning rate: 
    if count > halve:
        lr /= 2.0; count = 1 
    optimizer_w = optim.SGD([{'params': net_w.parameters()}], lr, momentum=0.9, weight_decay=wd)

    ### train with labeled data:    
    acc_l, loss_l = train_w(net_w, optimizer_w, nll_loss, trainloader_l)

    ### train with unlabeled data and SaaS estimates: 
    acc_u = train(net_w, optimizer_w, nll_loss, trainloader_u, yu)    
    
    ### run test to report the validation accuracy:
    prev_best_acc = best_acc
    valid_acc, best_acc, _ = test(net_w, nll_loss, testloader, best_acc)
    
    ### update the count:
    count = 1 if prev_best_acc < best_acc else count + 1    
    if lr < stop_lr:
        break
    
    print(y_file)
    print("epoch=%d,time=%f,lr=%f,acc_l=%f,acc_u=%f,valid_acc=%f,best_acc=%f" %\
          (epoch, time.time()-st_time, lr, acc_l, acc_u, valid_acc, best_acc))




