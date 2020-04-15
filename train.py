import torch
import torchvision
import torchvision.transforms as transforms
from enum import Enum
import os
from metrics import averager

class TrainingState(Enum):
    Epoch_Started = 1
    Epoch_Completed = 2
    Batch_Completed = 3    
    Iteration_Completed = 3    
    Training_Finished = 4
   
class Trainer(object):

    def __init__(self,criterion,optimizer,model,metrics=[],single_sample = False):
        self.criterion = criterion
        self.optimizer = optimizer
        self.model = model
        self.epoch = 0        
        self.current_batch_nr = 0
        self.total_batch_nr = 0
        self._observers = []
        self.metrics = metrics
        self.running_loss = averager()
        self.single_sample = single_sample
        return super().__init__()

    def register_observer(self, observer):
        self._observers.append(observer)
    
    def notify_observer(self, *args, **kwargs):
        for observer in self._observers:
            observer.notify(self, *args, **kwargs)

    def training_step(self,inputs, labels,use_gpu):        
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        # zero the parameter gradients
        self.optimizer.zero_grad()
        # # forward + backward + optimize
        outputs = self.model(inputs)

        #This computes the loss now twice, which is a bit unhandy
        for metric in self.metrics:
            metric.add(outputs.detach(),labels.detach())

        self.loss = self.criterion(outputs, labels)        
        self.loss.backward()

        self.running_loss.add(self.loss.item())

        self.optimizer.step()
        

    def log_loss(self, print_at, running_loss, writer):
        if self.current_batch_nr % print_at == print_at-1:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' % (self.epoch + 1, self.current_batch_nr + 1, running_loss.get_average()))                    
            #current index
            idx = self.epoch * self.total_batch_nr + self.current_batch_nr
            writer.add_scalar('Train/loss',self.loss.item(),idx)

    def run_epochs(self,trainloader,use_gpu,nr_epochs,writer = None):          
        
            print_at = 1000

            epoch_range = range(self.epoch,self.epoch+nr_epochs) 

            self.total_batch_nr = len(trainloader)

            for epoch in epoch_range:  # loop over the dataset multiple times

                self.epoch = epoch + 1

                self.notify_observer(TrainingState.Epoch_Started)
                
                self.running_loss.reset()
            
                for i, (inputs, labels) in enumerate(trainloader, 0):
                    
                    
                    self.current_batch_nr = i
                
                    self.training_step(inputs, labels,use_gpu)
                
                    #running_loss.add(self.loss.item())    
        
                    self.notify_observer(TrainingState.Batch_Completed)
                    #self.log_loss(print_at, running_loss, writer) 

                self.notify_observer(TrainingState.Epoch_Completed)

            self.notify_observer(TrainingState.Training_Finished)
            #writer.export_scalars_to_json('./all_Scalars.json')      


    #def save_model(model_name):
    #    torch.save(self.model, model_name)
    #    print("Succesfully saved model to %s" % model_name)

    #def load_model(model_path):
    #    net = torch.load(model_path)
    #    net.eval()
    #    return net

    def save_checkpoint(self,PATH):
        outdir  = os.path.join(PATH,"Checkpoints")
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        PATH  = os.path.join(outdir,"epoch{}.ckpt".format(self.epoch+1))
        torch.save({
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.loss            
                }, PATH)
        print("Succesfully saved model to %s" % PATH)
        return PATH

    def load_checkpoint(self,PATH):
        checkpoint = torch.load(PATH)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']
        print("Succesfully Loaded model from %s" % PATH)

class training_observer(object):
    def __init__(self, trainer):
        trainer.register_observer(self)        
        pass
    
    def notify(self, observable, training_state):
        if training_state is TrainingState.Batch_Completed:
            self.on_batch_completed(observable)
        elif training_state is TrainingState.Epoch_Completed:
            self.on_epoch_completed(observable)
        elif training_state is TrainingState.Epoch_Started:
            self.on_epoch_started(observable)
        elif training_state is TrainingState.Iteration_Completed:
            self.on_iteration_completed(observable)
        elif training_state is TrainingState.Training_Finished:
            self.on_training_finished(observable)        

    def on_batch_completed(self,trainer):
        pass

    def on_epoch_completed(self,trainer):
        pass

    def on_epoch_started(self,trainer):
        pass

    def on_iteration_completed(self,trainer):
        pass

    def on_training_finished(self,trainer):
        pass