#This document contains the training pipeline for a simple MNIST toy example
# This should not stay the same
import torch

import torch.nn as nn

from models import OwnSegNet

import metrics

import train

import test

import dl_logger

import dataloader

import numpy as np

import torchvision

import pandas as pd

class AnalysisBuilder(train.training_observer):

    '''Class to handle the analysis'''

    def __init__(self, trainer,tester,current_analysis_name,loggers,single_sample=False):
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
                    single_logger.log("Test/" + key,out_dict[key],epoch)                    
            out_dict = metric.get_per_class_accuracy()
            for key in out_dict.keys():
                for single_logger in self.loggers:
                    single_logger.log("Test/AccuracyClass" + key,out_dict[key],epoch) 
            conf_mat_dir = current_analysis_name + "/ConfusionMatrices"
            if not os.path.exists(conf_mat_dir):
                os.mkdir(conf_mat_dir)
            mat_path = os.path.join(conf_mat_dir,"ConfMat_epoch{}".format(epoch))
            np.save(mat_path,metric.conf_metric)
            pd.DataFrame(metric.conf_metric).to_csv(mat_path + ".csv")
            metric.reset()
    
    def on_loss_calulcated(self):
        step_size = 1
        
        
    def on_batch_completed(self,trainer):
        self.current_epoch = trainer.epoch
        step_size = 10
        if trainer.current_batch_nr % step_size == 0: 
            
            for single_logger in self.loggers:
                    single_logger.log("Train/" + "Loss",trainer.running_loss.get_average(),(trainer.epoch - 1) * trainer.total_batch_nr + trainer.current_batch_nr + 1)
            trainer.running_loss.reset()

            if not self.single_sample:
                for metric in trainer.metrics:
                    for single_logger in self.loggers:
                        single_logger.log("Train/" + metric.type(),metric.value(),(trainer.epoch - 1) * trainer.total_batch_nr + trainer.current_batch_nr + 1)
                    metric.reset()
                print("Batch {} completed".format(trainer.current_batch_nr))

                trainer.running_loss.reset()

    def on_epoch_completed(self,trainer):
        self.current_epoch = trainer.epoch
        use_gpu = next(trainer.model.parameters()).is_cuda         

        if not self.single_sample:

            step_size_save = 5
            if trainer.epoch % step_size_save == step_size_save - 1: 
                ckpt_name = trainer.save_checkpoint(self.current_name)                

            step_size = 10
            if trainer.epoch % step_size == step_size - 1: 
                
                tester.test_network(trainer.model,use_gpu,self.on_loss_calulcated,input_transform= input_transform,alpha=0.5)
                self.log_test_results(trainer.epoch)
        else:
            step_size = 3000
            if trainer.epoch % step_size == step_size - 1: 
                ckpt_name = trainer.save_checkpoint(self.current_name)
                tester.test_network(trainer.model,use_gpu,self.on_loss_calulcated,input_transform= input_transform,alpha=0.5)
                self.log_test_results(trainer.epoch)

        for single_logger in self.loggers:
            single_logger.close()        
        
        print("Epoch {} completed".format(trainer.epoch))


    def on_epoch_started(self,trainer):
        #self.current_epoch = trainer.epoch
        #tester.test_network(trainer.model,use_gpu,self.on_loss_calulcated)
        #self.log_test_results(trainer.epoch)
        print("Epoch {} started".format(trainer.epoch))
                    


import torch.optim as optim
from datetime import datetime

import os

from data import CamVid as camvid_dataset

def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)

def generate_folder_name(criterion, optimizer, single_sample,input_transform):
    '''Creates a folder name base on hyperparamters'''

    out_name = ""
    
    out_name += type(optimizer).__name__ + "_"
    
    
    for key in optimizer.param_groups[0].keys():
        if key not in 'params':
            out_name +=  key + "_" + str(optimizer.param_groups[0][key]) + "_"
    
    out_name += type(criterion).__name__    
    
    current_analysis_name = "run/" + out_name

    if input_transform == None:
        current_analysis_name += "_RGB"
    elif input_transform == ctf.madden:
        current_analysis_name += "_Madden"
    elif input_transform == ctf.madden_hs:
        current_analysis_name += "_Madden_HS"

    if single_sample:
        current_analysis_name += "_SingleSampleTest"
    return current_analysis_name



from matplotlib import pyplot as plt

import colortransforms as ctf

if __name__ == '__main__':    

    #Set to true to analyse a single batch
    #This will currently not produce the correct outputs, but it will help to see wether you converge
    single_sample = False

    #Get a reference to the directory where this file is locates
    outputdir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(outputdir)

    #Determine whether to use the GPU for training
    use_gpu = torch.cuda.is_available()

    # Set this to ctr.madden or ctr.madden_hs for the transform
    input_transform = None
    

    # Get input size of the netowk depeding on the color transform
    input_size = 3

    if input_transform == ctf.madden:
        input_size = 1
        
    net = OwnSegNet(input_size)        

    if use_gpu:
        net = net.cuda()
    
    #Load the Data
    loaders, w_class, class_encoding,sets = dataloader.get_data_loaders(camvid_dataset,4,4,4,single_sample=single_sample)
    trainloader, valloader, testloader = loaders  
    test_set,val_set,train_set = sets    

    lr = 1e-3
    wd = 5e-4 
    momentum = 0.9
      
    #Initialize the optimizer
    optimizer = optim.SGD(net.parameters(), lr=lr,weight_decay=wd,momentum=momentum)    

    ignore_index = list(class_encoding).index('unlabeled')

    #Initialize the criterion
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index,weight=w_class)        

    #Generate a name for the current analysis
    current_analysis_name = generate_folder_name(criterion, optimizer, single_sample,input_transform)    
        
    #Set the total number of epochs
    total_nr_epochs = 150

    #Load a checkpoint into the trainer if it already exists
    chkpt_path = current_analysis_name + "/Checkpoints"

    if os.path.exists(chkpt_path):
        latest_checkpoint = newest(chkpt_path)
        trainer.load_checkpoint(latest_checkpoint)
        trainer.epoch = int(os.path.basename(os.path.splitext(latest_checkpoint)[0])[5:])
        trainer.optimizer = optim.SGD(trainer.model.parameters(), lr=lr,weight_decay=wd,momentum=momentum)

    #Stop training if total nr epochs has been reached
    total_nr_epochs = total_nr_epochs - trainer.epoch
    print("Only {} epochs to go".format(total_nr_epochs))

    #Define metrics to be used during testing
    test_metrics = [metrics.ConfMatrix(len(class_encoding), ignore_index=ignore_index)]

    #Define ways to log the output during testing
    test_logger = [dl_logger.print_logger(),dl_logger.tensorboard_logger(current_analysis_name)]

    #Intialize the tester
    tester = test.Tester(testloader,criterion,test_metrics)

    #Feed eveything into the analysis builder which defines what to caluclate when and when to log
    analysis_builder = AnalysisBuilder(trainer,tester,current_analysis_name,test_logger,single_sample=single_sample)

    #Start the trainig
    trainer.run_epochs(trainloader,use_gpu,total_nr_epochs,input_transform= input_transform,alpha=0.5)
