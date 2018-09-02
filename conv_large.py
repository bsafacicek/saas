import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.cuda import FloatTensor

class conv_large(nn.Module):
    def __init__(self, is_bn):
        super(conv_large, self).__init__()
        self.is_bn = is_bn
        self.lr = nn.LeakyReLU(0.1)
        self.mp2_2 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop = nn.Dropout(p = 0.5)
        
        mom = 0.01 # update momentum, set it to 0 for keeping it fix
        is_aff = True
        
        self.bn_1 = nn.BatchNorm2d(128, eps=1e-06, momentum=mom, affine=is_aff)
        self.bn_2 = nn.BatchNorm2d(128, eps=1e-06, momentum=mom, affine=is_aff)
        self.bn_3 = nn.BatchNorm2d(128, eps=1e-06, momentum=mom, affine=is_aff)
        self.bn_4 = nn.BatchNorm2d(256, eps=1e-06, momentum=mom, affine=is_aff)
        self.bn_5 = nn.BatchNorm2d(256, eps=1e-06, momentum=mom, affine=is_aff)
        self.bn_6 = nn.BatchNorm2d(256, eps=1e-06, momentum=mom, affine=is_aff)
        self.bn_7 = nn.BatchNorm2d(512, eps=1e-06, momentum=mom, affine=is_aff)
        self.bn_8 = nn.BatchNorm2d(256, eps=1e-06, momentum=mom, affine=is_aff)
        self.bn_9 = nn.BatchNorm2d(128, eps=1e-06, momentum=mom, affine=is_aff)
        
        self.conv_1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False);                 
        self.conv_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False);
        self.conv_5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False);
        self.conv_6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False);
        self.conv_7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0, bias=False);
        self.conv_8 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False);
        self.conv_9 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False);

        self.avg_6 = nn.AvgPool2d(6, ceil_mode=True) #  average pooling
        self.fc = nn.Linear(128, 10)
        
    def forward(self, x, noise_std=0.0):

        #noise = Variable(FloatTensor(x.size()).normal_())*noise_std
        #x += noise   
        x = self.conv_1(x)
        if self.is_bn:
            x = self.bn_1(x) 
        x = self.lr(x)
        x = self.conv_2(x)
        if self.is_bn:
            x = self.bn_2(x)
        x = self.lr(x)     
        x = self.conv_3(x)
        if self.is_bn:
            x = self.bn_3(x)
        x = self.lr(x)  
        
        x = self.mp2_2(x)
        x = self.drop(x)
       
        x = self.conv_4(x)
        if self.is_bn:
            x = self.bn_4(x)
        x = self.lr(x)        
        x = self.conv_5(x)
        if self.is_bn:
            x = self.bn_5(x)
        x = self.lr(x)       
        x = self.conv_6(x)
        if self.is_bn:
            x = self.bn_6(x)
        x = self.lr(x)
        
        x = self.mp2_2(x)
        x = self.drop(x)
        
        x = self.conv_7(x)
        if self.is_bn:
            x = self.bn_7(x)
        x = self.lr(x)
        x = self.conv_8(x)
        if self.is_bn:
            x = self.bn_8(x)
        x = self.lr(x)        
        x = self.conv_9(x)
        if self.is_bn:
            x = self.bn_9(x)
        x = self.lr(x)  
        
        x = self.avg_6(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        
        return x
        

