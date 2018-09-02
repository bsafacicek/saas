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
import cvxpy
import argparse
np.set_printoptions(precision=3)
from saas_helper import scalar2onehot, unsup_nll, calc_entropy, setup

###  get dataset and network name as argument
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='svhn' or 'cifar10')
parser.add_argument('--net_name', help='conv_large' or 'resnet')
args = parser.parse_args()
dataset = args.dataset; net_name = args.net_name
assert(dataset in ['svhn', 'cifar10'])
assert(net_name in ['conv_large', 'resnet'])

### hyperparameters:
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

### load dataloaders:
if dataset == 'svhn':
    from datasets_svhn import get_loader_svhn, shuffle_loader_svhn
    trainloader_l, testloader, trainloader_u, trainset_l, test_set, trainset_u_org, lab_inds = \
    get_loader_svhn(nb_labelled, batch_size, unlab_rat, augment_type)
    train_labels_u_org = trainset_u_org.labels
    train_data_u_np = trainset_u_org.data
elif dataset == 'cifar10':   
    from datasets_cifar import get_loaders_cifar, shuffle_loader_cifar
    trainloader_l, testloader, trainloader_u, trainset_l, test_set, trainset_u_org, lab_inds = \
    get_loaders_cifar(nb_labelled, batch_size, unlab_rat, augment_type)
    train_labels_u_org = trainset_u_org.train_labels
    train_data_u_np = trainset_u_org.train_data

### prepare unlabeled data for pytorch setting:
train_labels_u_org = np.array(train_labels_u_org)
train_data_u = torch.from_numpy(np.float32(train_data_u_np)) 
train_data_u = train_data_u.permute(0,3,1,2)

### load or randomly initalize y_u estimates:
if y_file=='': # start y_u estimates randomly
    estimated_y = np.random.randint(0, 10,  len(train_labels_u_org))
    estimated_y = scalar2onehot(estimated_y)
else: # load y_u estimates from given .npy file
    estimated_y = np.load(y_file)   
estimated_y = Variable(torch.FloatTensor(estimated_y).cuda(), requires_grad=True)

st_time = time.time() 
for epoch in range(nb_outer_iter):    
    ### permute entire trainloader for unlabeled data and shuffling indexes to be used with estimated_y
    if dataset == 'svhn':        
        trainloader_u, ind_shuff_all = shuffle_loader_svhn(trainset_u_org, batch_size) 
    elif dataset == 'cifar10':   
        trainloader_u, ind_shuff_all = shuffle_loader_cifar(trainset_u_org, batch_size) 
           
    ### start with random weights at the beggining of each outer epoch:
    net_w = copy.deepcopy(net_w_orig) 
    net_w.train()
    optimizer_w = optim.SGD([{'params': net_w.parameters()}], eta_w, momentum=0.9, weight_decay=0)
    
    ### argmax the current posterior to plot y_u accurucies:
    est_y_arg = np.array(np.argmax(estimated_y.data.cpu().numpy(), axis=1))
    y_acc = ((est_y_arg==np.array(train_labels_u_org)).sum())*1.0/len(train_labels_u_org)    
    print(file_name)
    print("epoch=%d,time=%f,y_acc=%f" % (epoch, time.time()-st_time, y_acc))
    st_time = time.time()
    
    for epoch_w in range(nb_inner_iter):
        for batch_idx, (inps_u, targs_u) in enumerate(trainloader_u):
            ### load labeled batches:
            inps_l, targs_l = iter(trainloader_l).next()
            inps_l, targs_l = Variable(inps_l.cuda()), Variable(targs_l.cuda())
            outs_l = logsoft(net_w(inps_l))
            ind_shuff = copy.deepcopy(ind_shuff_all[batch_idx*batch_size:(batch_idx+1)*batch_size])
            loss_l = nll_loss(outs_l, targs_l)
            
            ### extract the current label estimates for the batch:
            est_y = estimated_y[ind_shuff]
            
            ### calculate the losses for unlabeled data:
            inps_u = Variable(inps_u.cuda())
            outs_u = net_w(inps_u)
            loss_ent = lambda_ent*calc_entropy(outs_u)
            outs_u = logsoft(outs_u)
            loss_u = unsup_nll_loss(est_y, 0, outs_u)  
            
            ### sum all the losses:
            loss = loss_l + loss_u + loss_ent 
                
            ### Add Gaussian noise to weights:      
            for param in net_w.parameters():
                noise = torch.cuda.FloatTensor(param.size()).normal_()
                param.data += ((eta_w*langevin_coef)**0.5)*noise
               
            ### update weights:
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
    state = {"lab_inds": lab_inds, "estimated_y": estimated_y.data.cpu().numpy(), "epoch": epoch, "y_acc": y_acc,}
    np.save(y_u_folder+file_name+'.npy', state) 
    



