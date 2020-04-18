import torch

import torch.nn as nn

from models import OwnSegNet

from models import SegNet

import metrics

import train

import test

import dl_logger

import dataloader

import numpy as np

import torchvision

import pandas as pd


from matplotlib import pyplot as plt

import colortransforms as ctf

import torch.optim as optim
from datetime import datetime

import os

from data import CamVid as camvid_dataset

def image_tensor_to_image(img):
    
    to_pil = torchvision.transforms.ToPILImage()
    pil_img = to_pil(img)

    return pil_img
   

def label_tensor_to_image(lab,class_encoding):

    to_pil = torchvision.transforms.ToPILImage()

    out_img = torch.zeros([3,lab.shape[0],lab.shape[1]])

    for j,k in enumerate(class_encoding):
        # Get all indices for current class
        pos = torch.where(lab == j)
        col = class_encoding[k]
        out_img[0,pos[0],pos[1]] = col[0]
        out_img[1,pos[0],pos[1]] = col[1]
        out_img[2,pos[0],pos[1]] = col[2]
                                
    
    pil_img = to_pil(out_img)
    return pil_img

if __name__ == '__main__':    

    single_sample = True

    outputdir = os.path.dirname(os.path.abspath(__file__))

    os.chdir(outputdir)

    use_gpu = torch.cuda.is_available()

    own_net = SegNet(3,12)#OwnSegNet(3)

        
    loaders, w_class, class_encoding,sets = dataloader.get_data_loaders(camvid_dataset,1,1,1,single_sample=single_sample)
    trainloader, valloader, testloader = loaders  
    test_set,val_set,train_set = sets
    
    for i,key in enumerate(class_encoding.keys()):
        print("{} \t {}".format(i,key))

    optimizer = optim.SGD(own_net.parameters(), lr=1e-3,weight_decay=5e-4,momentum=0.9)
    
    # Evaluation metric

    ignore_index = list(class_encoding).index('unlabeled')
    
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)   

    if use_gpu:
        own_net = own_net.cuda()
    
    trainer = train.Trainer(criterion,optimizer,own_net,single_sample=single_sample)


    chkpt_path = "D:\CloudStorage\ODWork\Work Dropbox\MB\Tmp\SGD_lr_0.001_momentum_0.9_dampening_0_weight_decay_0.0005_nesterov_False_CrossEntropyLoss_RGB_2\Checkpoints\epoch150.ckpt"
    
    trainer.load_checkpoint(chkpt_path)
    
    own_net = trainer.model

    own_net.eval()
    

    for i, (inputs, labels) in enumerate(trainloader, 0):

        labels = labels.to(dtype=torch.long)

        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        
        # # forward + backward + optimize
        raw,outputs = own_net(inputs)                

        _,pred = outputs.max(dim=1)

        image_tensor_to_image(inputs.squeeze(0).cpu()).save("ImageExample.bmp")
        label_tensor_to_image(labels.squeeze(0).cpu(),class_encoding).save("LabelExample.bmp")
        label_tensor_to_image(pred.squeeze(0).cpu(),class_encoding).save("PredictionExample.bmp")
        
                
