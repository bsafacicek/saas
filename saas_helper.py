import torch
import torch.nn as nn
import torch.nn.init as init
soft = nn.Softmax() if torch.__version__[2]=="1" else nn.Softmax(dim=1)
logsoft = nn.LogSoftmax() if torch.__version__[2]=="1" else nn.LogSoftmax(dim=1)

import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # for server use
import matplotlib.pyplot as plt
plt.ioff() #non-interactive mode delays all drawing until show() is called

import torch
import torch.nn as nn
from torch.autograd import Variable

class unsup_nll(torch.nn.Module):
    # N: number of unlabelled samples
    # n: number of classes
    def __init__(self, batch_size):
        super(unsup_nll, self).__init__()
        self.batchsize = batch_size
         
    # softmax_outs: (batchsize, nb_classes)
    def forward(self, est_y, bn, logsoftmax_outs):
        cur_batchsize = len(logsoftmax_outs)
        est_labels = est_y[bn*self.batchsize:bn*self.batchsize+cur_batchsize,:]
        return -torch.sum(est_labels*logsoftmax_outs[:,0:10])/cur_batchsize

    # vec: (N, 1) cuda FloatTensor Variable  probability vector
    def calc_entropy(self, vec):
        return -torch.dot(vec, torch.log(vec+10**-8)) 

def eucl_dist(outp1, outp2):
    return nn.functional.pairwise_distance(outp1, outp2)

def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, 0, mode='fan_in')
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3) 
            
def calc_entropy(vecs):
    vecs = soft(vecs)
    loss_ent = -torch.mean(torch.sum(vecs*torch.log(vecs+10**-8), 1)) 
    return loss_ent

def scalar2onehot(labels, n=10):
    leng = len(labels)
    labels_hot = np.ones((leng, n))
    for i, yl in enumerate(labels):
        labels_hot[i, yl] = 9
        labels_hot[i, :] /= 1.0*np.sum(labels_hot[i, :])
    return labels_hot

def setup(dataset, net_name, paths=[]):

    if dataset == 'svhn':
        nb_labelled = 1000
    elif dataset == 'cifar10':
        nb_labelled = 4000
        
    if dataset == 'svhn':
        nb_outer_iter = 75 # number of outer loop iterations
        nb_inner_iter = 5 # number of inner loop iterations
    elif dataset == 'cifar10':
        nb_outer_iter = 135 # number of outer loop iterations
        nb_inner_iter = 40 # number of inner loop iterations
        
    if net_name == "resnet":
        from resnet import ResNet18
        net_w_orig = ResNet18().cuda()
        augment_type = "affine" # uses affine transformations as data augmentations
    elif net_name == "conv_large":    
        from conv_large import conv_large
        is_bn = True
        net_w_orig = conv_large(is_bn).cuda()
        augment_type = "mean" # "mean" uses the augmentations used in previous SSL papers. 
        
    ### create the directories:
    for path in paths:
        if not os.path.exists(path): 
            os.makedirs(path)
            
    return net_w_orig, augment_type, nb_labelled, nb_inner_iter, nb_outer_iter

def train_w(net, optimizer, criterion, trainloader):
    net.train()
    train_loss, correct, total = 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
        outputs = net(inputs)
        outputs = logsoft(outputs)
        loss = criterion(outputs, targets)        
        loss.backward(); optimizer.step(); optimizer.zero_grad()

        train_loss += loss.data[0]
        confs, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
            
    return 100.*correct/total, train_loss/(batch_idx+1)

def test(net, criterion, testloader, best_acc):
    net.eval()
    correct, total, test_loss = 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
        outputs = net(inputs)
        outputs = logsoft(outputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc

    return acc, best_acc, test_loss/(batch_idx+1)