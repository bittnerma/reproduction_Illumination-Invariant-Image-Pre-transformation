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

class AnalysisBuilder(train.training_observer):
    def __init__(self, trainer,tester,current_analysis_name,loggers):
        super(AnalysisBuilder,self).__init__(trainer)               
        
        self.current_name = current_analysis_name
        self.loggers = loggers
        self.current_epoch = 0

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
        step_size = 3
        if trainer.current_batch_nr % step_size == 0: 
            
            for single_logger in self.loggers:
                    single_logger.log("Train/"+"Loss",trainer.running_loss.get_average(),(trainer.epoch-1) * trainer.total_batch_nr + trainer.current_batch_nr)
            trainer.running_loss.reset()

            for metric in trainer.metrics:
                for single_logger in self.loggers:
                    single_logger.log("Train/"+metric.type(),metric.value(),(trainer.epoch-1) * trainer.total_batch_nr + trainer.current_batch_nr)
                metric.reset()
            print("Batch {} completed".format(trainer.current_batch_nr))

            trainer.running_loss.reset()

    def on_epoch_completed(self,trainer):
        self.current_epoch = trainer.epoch
        use_gpu = next(trainer.model.parameters()).is_cuda         

        tester.test_network(trainer.model,use_gpu,self.on_loss_calulcated)
        self.log_test_results(trainer.epoch)

        step_size = 5
        if trainer.epoch % step_size == step_size-1: 
            ckpt_name = trainer.save_checkpoint(self.current_name)

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

if __name__ == '__main__':    

    outputdir = os.path.dirname(os.path.abspath(__file__))

    os.chdir(outputdir)

    use_gpu = torch.cuda.is_available()

    net = OwnSegNet(3)

    if use_gpu:
        net = net.cuda()

    #trainloader,testloader,classes = dataloader.load_MNIST(4)

    loaders, w_class, class_encoding = dataloader.get_data_loaders(camvid_dataset,2,2,2)
    trainloader, valloader, testloader = loaders

    # optimizer = optim.Adam(net.parameters(), lr=0.0005,betas=(0.9,0.99))

    # criterion = nn.CrossEntropyLoss()
    
    #As in the paper
    optimizer = optim.SGD(net.parameters(), lr=1e-3,weight_decay=5e-4,momentum=0.9)
    #optimizer = optim.Adam(net.parameters(), lr=1e-3,betas=(0.9,0.99),weight_decay=5e-4)
    
        # Evaluation metric

    ignore_index = list(class_encoding).index('unlabeled')

     #TODO: Find the right index
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index,weight=w_class)    

    out_name = ""

    out_name += type(optimizer).__name__ + "_"
    

    for key in optimizer.param_groups[0].keys():
        if key not in 'params':
            out_name +=  key+"_"+str(optimizer.param_groups[0][key]) + "_"

    out_name += type(criterion).__name__ 

    #current_analysis_name = "run/"+datetime.now().strftime("%d%m%Y%H%M%S")     
    current_analysis_name = "run/"+out_name + "RGB"

    
    nr_epochs= 370
        
    trainer = train.Trainer(criterion,optimizer,net)

    chkpt_path = current_analysis_name+"/Checkpoints"

    if os.path.exists(chkpt_path):
        latest_checkpoint = newest(chkpt_path)
        trainer.load_checkpoint(latest_checkpoint)
        trainer.epoch = int(os.path.basename(os.path.splitext(latest_checkpoint)[0])[5:])
        optimizer = optim.SGD(trainer.model.parameters(), lr=1e-1,weight_decay=5e-4,momentum=0.9)

    test_metrics = [metrics.ConfMatrix(len(class_encoding), ignore_index=ignore_index)]

    test_logger = [dl_logger.print_logger(),dl_logger.tensorboard_logger(current_analysis_name)]

    tester = test.Tester(testloader,criterion,test_metrics)

    analysis_builder = AnalysisBuilder(trainer,tester,current_analysis_name,test_logger)

    trainer.run_epochs(trainloader,use_gpu,nr_epochs)

    #trainer.load_checkpoint(ckpt_name)

    #trainer.run_epochs(trainloader,use_gpu,nr_epochs,writer)

  
