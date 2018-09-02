import numpy as np
import copy
import cv2

import torch
from torch.utils.data import  DataLoader
import torchvision
from datasets_cifar import noaug_cifar10

    
def zca_vat(data, eps=500,is_zca=True):
    from scipy import linalg
    mean = np.mean(data, axis=0)
    mdata = data - mean
    sigma = np.dot(mdata.T, mdata) / mdata.shape[0]
    U, S, V = linalg.svd(sigma)

    if is_zca:
        components = np.dot(np.dot(U, np.diag((np.array(S))**(-0.5) + eps)), V)
    else:
        components = np.dot(np.diag(1 / np.sqrt(S) + eps), U.T)
    whiten = np.dot(data - mean, components.T)

    return components, mean, whiten


def whiten_cifar10(file_zca):
    # load original cifar10 data:
    trainset = CIFAR10(root='./data', train=True, download=False)
    testset = CIFAR10(root='./data', train=False, download=False)
    train_x = trainset.train_data.astype('float32')
    test_x = testset.test_data.astype('float32')
    train_x_norm = (train_x - 127.5) / 255.; test_x_norm = (test_x - 127.5) / 255.

    # vectorize inputs before whitening:
    train_x_vec = np.reshape(train_x_norm, (train_x_norm.shape[0], -1))
    test_x_vec = np.reshape(test_x_norm, (test_x_norm.shape[0], -1))
    
    # learn ZCA parameters from training data:
    components, mean, train_x_zca = zca_vat(train_x_vec)

    # apply ZCA to test data:
    test_x_zca = np.dot(test_x_vec - mean, components.T)

    # reshape back to original image shape:
    train_x_zca = np.reshape(train_x_zca, train_x.shape)
    test_x_zca = np.reshape(test_x_zca, test_x.shape) 
    
    np.savez(file_zca, train_x_zca=train_x_zca, test_x_zca=test_x_zca)

class CIFAR10(torchvision.datasets.CIFAR10):
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


def get_loaders_cifar_zca(nb_labelled, batch_size, is_whiten, file_zca, lab_inds=[]):
    x_zca = np.load(file_zca); train_x_zca = x_zca['train_x_zca']; test_x_zca = x_zca['test_x_zca']

    transform_train, transform_test = noaug_cifar10() 
    trainset_l = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_set = CIFAR10(root='./data', train=False, download=True, transform=transform_test)        
    
    if is_whiten:
        ### Replace with whitened data
        trainset_l.train_data = copy.deepcopy(train_x_zca)
        test_set.test_data = copy.deepcopy(test_x_zca) 

    if len(lab_inds) == 0: 
        lab_inds = []
        for i in range(10):    
            labels = np.array(trainset_l.train_labels) 
            inds_i = np.where(labels == i)[0]
            inds_i = np.random.permutation(inds_i)
            lab_inds.extend(inds_i[0:int(nb_labelled/10)].tolist())
        lab_inds = np.array(lab_inds)
    
    all_inds = np.arange(len(trainset_l.train_labels))
    unlab_inds = np.setdiff1d(all_inds, lab_inds)    
    
    trainset_u = copy.deepcopy(trainset_l)    
    trainset_u.train_data = np.array(trainset_u.train_data)[unlab_inds]
    trainset_u.train_labels = np.array(trainset_u.train_labels)[unlab_inds]
    trainloader_u = DataLoader(trainset_u, batch_size=batch_size, shuffle=False, num_workers=1)
    print (trainset_u.train_data.shape, len(trainset_u.train_labels))

    trainset_l.train_data = np.array(trainset_l.train_data)[lab_inds] 
    trainset_l.train_labels = np.array(trainset_l.train_labels)[lab_inds]            
    print (trainset_l.train_data.shape, len(trainset_l.train_labels))
    trainloader_l = DataLoader(trainset_l, batch_size=batch_size, shuffle=True, num_workers=1)
    
    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1) 

    return trainloader_l, testloader, trainloader_u, trainset_l, test_set, trainset_u, lab_inds
    

class translate_flip:

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_images):
        old_images = old_images.numpy()
        if np.random.randint(2, size=1)[0] == 0:
            old_images = np.flip(old_images, axis=2)

        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)

        N,  xsize, ysize, _  = old_images.shape
        new_images = np.zeros((N, xsize+2*xpad, ysize+2*ypad, 3), dtype=np.float32)

        if np.random.randint(2, size=1)[0] == 0: # reflect
            for i, old_image in enumerate(old_images):
                new_images[i] = cv2.copyMakeBorder(old_image,xpad,xpad,ypad,ypad,cv2.BORDER_REFLECT) # top, bottom, left, right
        else:  #  zero padding
            new_images[:, xpad:xpad+xsize,ypad:ypad+ysize,:] = old_images #original image into the middle of the padded image
        
        new_images = new_images[:, xpad - xtranslation:xpad + xsize - xtranslation, 
                              ypad - ytranslation:ypad + ysize - ytranslation]   # crop
 
        new_images = torch.from_numpy(new_images)

        return new_images    


