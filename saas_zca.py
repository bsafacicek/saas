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
import os
import cvxpy
np.set_printoptions(precision=3)
    
from zca import get_loaders_cifar_zca, whiten_cifar10
from saas_helper import scalar2onehot, unsup_nll, calc_entropy, setup

file_zca = "data/cifar10_whitened.npz"
if not os.path.exists(file_zca):             
    whiten_cifar10(file_zca)

### hyperparameters:
dataset = 'cifar10' 
net_name = 'conv_large' 
lambda_ent = 1.0 # weight of entropy in the overall loss
eta_w =  0.01 # learning rate for weights in the inner loop
eta_y = 1.0 # learning rate for estimated labels in the outer loop
epsilon = 0.05 # label threshold
batch_size = 100 # batch size for both labeled and unlabeled data
unlab_rat = 1.0 # Percentage of the unlabeled data to be used. If 1.0, entire training data is used. 
langevin_coef = 10**-5 # determines the deviation of the noise added to weigths
plot_folder = 'plots/' # folder to save the plot of y_u accuracies in time
y_u_folder = 'estimated_labels/' # folder to save the y_u estimates
y_file = '' # path to the .npy file from which y_u loaded. If "", start with random estimates.

### setup directories and set dataset/net_name dependent parameters:
net_w_orig, augment_type, nb_labelled, nb_inner_iter, nb_outer_iter = setup(dataset, net_name, [plot_folder, y_u_folder])
file_name = '%s_%s'%(dataset, net_name)
unsup_nll_loss = unsup_nll(batch_size) 
nb_unlabelled = 50000-nb_labelled
is_whiten = True

### load dataloaders:
_, _, _, trainset_l, test_set, trainset_u, lab_inds = \
get_loaders_cifar_zca(nb_labelled, batch_size, is_whiten, file_zca)

max_batch_idx_l = nb_labelled/batch_size 
 
train_labels_u = np.array(trainset_u.train_labels)
train_data_u = torch.from_numpy(np.float32(trainset_u.train_data)) 
train_labels_l = np.array(trainset_l.train_labels)
train_data_l = torch.from_numpy(np.float32(trainset_l.train_data))     
train_labels_u= torch.from_numpy(train_labels_u)
train_labels_l= torch.from_numpy(train_labels_l)

### load or randomly initalize y_u estimates:
if y_file=='': # start y_u estimates randomly
    estimated_y = np.random.randint(0, 10,  len(train_labels_u))
    estimated_y = scalar2onehot(estimated_y)
else: # load y_u estimates from given .npy file
    estimated_y = np.load(y_file)   
estimated_y = Variable(torch.FloatTensor(estimated_y).cuda(), requires_grad=True)
    
def augment(x):
    if augment_type == "mean":
        from zca import translate_flip
        obj = translate_flip(4)     
        return obj(x)
    else:
        assert NotImplementedError

st_time = time.time()        
for epoch in range(nb_outer_iter):    
    
    ### start with random weights at the beggining of each outer epoch:
    net_w = copy.deepcopy(net_w_orig) 
    net_w.train()
    optimizer_w = optim.SGD([{'params': net_w.parameters()}], eta_w, momentum=0.9, weight_decay=0)
               
    ### argmax the current posterior to plot y_u accurucies:
    est_y_arg = np.array(np.argmax(estimated_y.data.cpu().numpy(), axis=1))
    y_acc = ((est_y_arg==np.array(train_labels_u)).sum())*1.0/len(train_labels_u)    
    print(file_name)
    print("epoch=%d,time=%f,y_acc=%f" % (epoch, time.time()-st_time, y_acc))
    st_time = time.time()
    
    ### permurtation indexes for labeled and unlabeled data
    inds_all_l = np.arange(nb_labelled)
    ind_shuff_all_l = np.random.permutation(inds_all_l)
    inds_all_u = np.arange(nb_unlabelled)
    ind_shuff_all_u = np.random.permutation(inds_all_u)
    
    for epoch_w in range(nb_inner_iter):
        for batch_idx in range(nb_unlabelled/batch_size):
            ### shuffle and augment labeled batch
            batch_idx_l = batch_idx % max_batch_idx_l
            ind_shuff_l = copy.deepcopy(ind_shuff_all_l[batch_idx_l*batch_size:(batch_idx_l+1)*batch_size])
            inps_l = copy.deepcopy(train_data_l[ind_shuff_l])
            inps_l = augment(inps_l)
            inps_l = inps_l.permute(0,3,1,2)
            targs_l =  copy.deepcopy(train_labels_l[ind_shuff_l])     
            inps_l, targs_l = Variable(inps_l.cuda()), Variable(targs_l.cuda())
            outs_l = logsoft(net_w(inps_l))
            loss_l = nll_loss(outs_l, targs_l)

            ### shuffle and augment unlabeled batch
            ind_shuff_u = copy.deepcopy(ind_shuff_all_u[batch_idx*batch_size:(batch_idx+1)*batch_size])
            inps_u = copy.deepcopy(train_data_u[ind_shuff_u])
            inps_u = augment(inps_u)
            inps_u = inps_u.permute(0,3,1,2)
            targs_u =  copy.deepcopy(train_labels_u[ind_shuff_u])          
            inps_u, targs_u = Variable(inps_u.cuda()), Variable(targs_u.cuda())

            ### extract the current label estimates for the batch:
            est_y = estimated_y[ind_shuff_u]
            
            ### calculate the losses for unlabeled data:
            outs_u = net_w(inps_u)
            loss_ent = lambda_ent*calc_entropy(outs_u)
            outs_u = logsoft(outs_u)
            loss_u = unsup_nll_loss(est_y, 0, outs_u)  
            loss = loss_l + loss_u + loss_ent 

            ### Add Gaussian noise to weights:          
            for param in net_w.parameters():
                noise = torch.cuda.FloatTensor(param.size()).normal_()
                param.data += ((eta_w*langevin_coef)**0.5)*noise
                              
            loss.backward(); optimizer_w.step(); net_w.zero_grad()

    ### update unlabel estimates with cumulative loss:
    optimizer_y = optim.SGD([{'params': estimated_y, 'lr':eta_y}],  momentum=0, weight_decay=0)
    optimizer_y.step(); optimizer_y.zero_grad(); estimated_y.grad.data.zero_()

    ### Project onto probability polytope
    est_y_num = estimated_y.data.cpu().numpy()
    U = cvxpy.Variable(est_y_num.shape[0], est_y_num.shape[1])
    objective = cvxpy.Minimize(cvxpy.sum_entries(cvxpy.square(U - est_y_num)))
    constraints = [U >= epsilon, cvxpy.sum_entries(U, axis=1) == 1]
    prob = cvxpy.Problem(objective, constraints)
    prob.solve()
    estimated_y.data = torch.from_numpy(np.float32(U.value)).cuda()
     
    ### save estimated labels and labeled indexes on unlabeled data:
    state = {"lab_inds": lab_inds, "estimated_y": estimated_y.data.cpu().numpy(), "epoch":epoch, "y_acc":y_acc}
    np.save(y_u_folder+file_name+'.npy', state)
    
    

