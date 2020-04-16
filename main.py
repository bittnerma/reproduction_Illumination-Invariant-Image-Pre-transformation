#This document contains the training pipeline for a simple MNIST toy example
# This should not stay the same 
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

class AnalysisBuilder(train.training_observer):
    def __init__(self, trainer,tester,current_analysis_name,loggers,single_sample = False):
        super(AnalysisBuilder,self).__init__(trainer)               
        
        self.current_name = current_analysis_name
        self.loggers = loggers
        self.current_epoch = 0
        self.single_sample = single_sample

    def log_test_results(self, epoch):
        for metric in tester.metrics:
            out_dict = metric.get_mean_metrics()
            for key in out_dict.keys():
                for single_logger in self.loggers:
                    single_logger.log("Test/"+key,out_dict[key],epoch)                    
            out_dict = metric.get_per_class_accuracy()
            for key in out_dict.keys():
                for single_logger in self.loggers:
                    single_logger.log("Test/AccuracyClass"+key,out_dict[key],epoch) 
            conf_mat_dir = current_analysis_name+"/ConfusionMatrices"
            if not os.path.exists(conf_mat_dir):
                os.mkdir(conf_mat_dir)
            mat_path = os.path.join(conf_mat_dir,"ConfMat_epoch{}".format(epoch))
            np.save(mat_path,metric.conf_metric)
            metric.reset()

    #Maybe it's sufficient to calculate this at the same time as everything else
    def on_loss_calulcated(self):
        step_size = 1
        # if tester.current_batch_nr % step_size == 0: 
        #     for single_logger in self.loggers:
        #         single_logger.log("Test/Loss",tester.running_loss.get_average(),(self.current_epoch-1) * tester.total_batch_nr + tester.current_batch_nr) 
        #     tester.running_loss.reset()
        
    def on_batch_completed(self,trainer):
        self.current_epoch = trainer.epoch
        step_size = 10
        if trainer.current_batch_nr % step_size == 0: 
            
            for single_logger in self.loggers:
                    single_logger.log("Train/"+"Loss",trainer.running_loss.get_average(),(trainer.epoch-1) * trainer.total_batch_nr + trainer.current_batch_nr + 1)
            trainer.running_loss.reset()

            if not self.single_sample:
                for metric in trainer.metrics:
                    for single_logger in self.loggers:
                        single_logger.log("Train/"+metric.type(),metric.value(),(trainer.epoch-1) * trainer.total_batch_nr + trainer.current_batch_nr + 1)
                    metric.reset()
                print("Batch {} completed".format(trainer.current_batch_nr))

                trainer.running_loss.reset()

    def on_epoch_completed(self,trainer):
        self.current_epoch = trainer.epoch
        use_gpu = next(trainer.model.parameters()).is_cuda         

        if not self.single_sample:

            step_size_save = 5
            if trainer.epoch % step_size_save == step_size_save-1: 
                ckpt_name = trainer.save_checkpoint(self.current_name)                

            step_size = 10
            if trainer.epoch % step_size == step_size-1: 
                
                tester.test_network(trainer.model,use_gpu,self.on_loss_calulcated)
                self.log_test_results(trainer.epoch)
        else:
            step_size = 3000
            if trainer.epoch % step_size == step_size-1: 
                ckpt_name = trainer.save_checkpoint(self.current_name)
                tester.test_network(trainer.model,use_gpu,self.on_loss_calulcated)
                self.log_test_results(trainer.epoch)

        for single_logger in self.loggers:
            single_logger.close()        
        
        print("Epoch {} completed".format(trainer.epoch))


    def on_epoch_started(self,trainer):
        #self.current_epoch = trainer.epoch
        tester.test_network(trainer.model,use_gpu,self.on_loss_calulcated)
        self.log_test_results(trainer.epoch)
        print("Epoch {} started".format(trainer.epoch))
                    


import torch.optim as optim
from datetime import datetime

import os

from data import CamVid as camvid_dataset

def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)

def generate_folder_name(criterion, optimizer, single_sample):
    out_name = ""
    
    out_name += type(optimizer).__name__ + "_"
    
    
    for key in optimizer.param_groups[0].keys():
        if key not in 'params':
            out_name +=  key+"_"+str(optimizer.param_groups[0][key]) + "_"
    
    out_name += type(criterion).__name__ 
    
    
    
    #current_analysis_name = "run/"+datetime.now().strftime("%d%m%Y%H%M%S")     
    current_analysis_name = "run/"+out_name + "_RGB" + "_OtherNet"
    
    if single_sample:
        current_analysis_name += "_SingleSampleTest"
    return current_analysis_name

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

from matplotlib import pyplot as plt

if __name__ == '__main__':    

    single_sample = False

    outputdir = os.path.dirname(os.path.abspath(__file__))

    os.chdir(outputdir)

    use_gpu = torch.cuda.is_available()

    #net = OwnSegNet(3)
    net = SegNet(3,12)

    if use_gpu:
        net = net.cuda()
    trainer = None
    
    loaders, w_class, class_encoding,sets = dataloader.get_data_loaders(camvid_dataset,4,4,4,single_sample=single_sample)
    trainloader, valloader, testloader = loaders  
    test_set,val_set,train_set = sets

    label_tensor_to_image(test_set[0][1],class_encoding)

    lr = 1e-3
    wd = 5e-4 #Turning off regularization for debugging
    momentum = 0.9

    #As in the paper
    optimizer = optim.SGD(net.parameters(), lr=lr,weight_decay=wd,momentum=momentum)
    
    # Evaluation metric

    ignore_index = list(class_encoding).index('unlabeled')
    
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index,weight=w_class)        

    current_analysis_name = generate_folder_name(criterion, optimizer, single_sample)    
    
    nr_epochs= 435
        
    if trainer is None:
        trainer = train.Trainer(criterion,optimizer,net,single_sample=single_sample)
    else:
        trainer.optimizer = optimizer
        trainer.criterion = criterion
        trainer.net = net
        trainer.epoch = 0
        trainer.running_loss.reset()
        

    chkpt_path = current_analysis_name+"/Checkpoints"

    if os.path.exists(chkpt_path):
        latest_checkpoint = newest(chkpt_path)
        trainer.load_checkpoint(latest_checkpoint)
        trainer.epoch = int(os.path.basename(os.path.splitext(latest_checkpoint)[0])[5:])
        trainer.optimizer = optim.SGD(trainer.model.parameters(), lr=lr,weight_decay=wd,momentum=momentum)

    test_metrics = [metrics.ConfMatrix(len(class_encoding), ignore_index=ignore_index)]

    test_logger = [dl_logger.print_logger(),dl_logger.tensorboard_logger(current_analysis_name)]

    tester = test.Tester(testloader,criterion,test_metrics)

    analysis_builder = AnalysisBuilder(trainer,tester,current_analysis_name,test_logger,single_sample=single_sample)

    trainer.run_epochs(trainloader,use_gpu,nr_epochs,class_encoding=class_encoding)

        #trainer.load_checkpoint(ckpt_name)

        #trainer.run_epochs(trainloader,use_gpu,nr_epochs,writer)

  
