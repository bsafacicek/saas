import numpy as np
import copy

import torch
from torch.autograd import Variable
import torch.nn as nn

logsoft = nn.LogSoftmax()

from zca import translate_flip
def augment(x):
    obj = translate_flip(4)     
    return obj(x)

def train_w(net, optimizer, criterion, inp_all, targ_all, batch_size):
    net.train()
    train_loss, correct, total = 0, 0, 0
    nb_labelled = len(targ_all)
    inds_all = np.arange(nb_labelled)
    inp_all = np.float32(inp_all)

    ind_shuff_all = np.random.permutation(inds_all)
    for batch_idx in range(nb_labelled/batch_size):   
        ind_shuff = copy.deepcopy(ind_shuff_all[batch_idx*batch_size:(batch_idx+1)*batch_size])
        inputs = copy.deepcopy(inp_all[ind_shuff])
        targets =  copy.deepcopy(targ_all[ind_shuff])   
        
        inputs = torch.from_numpy(inputs)
        inputs = augment(inputs)        
        inputs = inputs.permute(0,3,1,2)
        targets = torch.from_numpy(targets)
        optimizer.zero_grad()
        inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
        outputs = net(inputs)
        outputs = logsoft(outputs)
        
        loss = criterion(outputs, targets)        
    
        loss.backward(); optimizer.step()
        
        train_loss += loss.data[0]
        confs, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
    return 100.*correct/total, train_loss/(batch_idx+1) 
    
    
def train_yu(net, optimizer, criterion, inp_all, label_org, est_ys, batch_size):  
    net.train()
    train_loss, correct, total = 0, 0, 0
    nb_labelled = len(est_ys)
    inds_all = np.arange(nb_labelled)
    inp_all = np.float32(inp_all)

    ind_shuff_all = np.random.permutation(inds_all)
    for batch_idx in range(nb_labelled/batch_size):   
        ind_shuff = copy.deepcopy(ind_shuff_all[batch_idx*batch_size:(batch_idx+1)*batch_size])
        inputs = copy.deepcopy(inp_all[ind_shuff])
        targets =  copy.deepcopy(est_ys[ind_shuff])   
        inputs = torch.from_numpy(inputs)
        inputs = augment(inputs)        
        inputs = inputs.permute(0,3,1,2)
        inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
        outputs = logsoft(net(inputs))
        loss = criterion(outputs, targets)            
        loss.backward(); optimizer.step(); optimizer.zero_grad()

        confs, predicted = torch.max(outputs.data, 1)
        
        # get correct labels for post mortem:
        targs_u_org = copy.deepcopy(label_org[ind_shuff])         
        targs_u_org = torch.from_numpy(targs_u_org) 
        
        train_loss += loss.data.cpu().numpy()[0]        
        total += targs_u_org.size(0)
        correct += predicted.cpu().eq(targs_u_org).sum()
        
    return 100.*correct/total, train_loss/(batch_idx+1) 
        
def test(net, inp_all, targ_all, best_acc, batch_size):
    net.eval()
    correct, total = 0, 0
    nb_labelled = len(targ_all)
    inp_all = np.array(inp_all); targ_all = np.array(targ_all)
    inp_all = np.float32(inp_all)

    for batch_idx in range(nb_labelled/batch_size):   
        inputs = copy.deepcopy(inp_all[batch_idx*batch_size:(batch_idx+1)*batch_size])
        targets =  copy.deepcopy(targ_all[batch_idx*batch_size:(batch_idx+1)*batch_size])
        inputs = torch.from_numpy(inputs)
        inputs = inputs.permute(0,3,1,2)

        targets = torch.from_numpy(targets)
        inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
        outputs = net(inputs)
        outputs = logsoft(outputs)
        
        confs, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
            
    return acc, best_acc 
    

