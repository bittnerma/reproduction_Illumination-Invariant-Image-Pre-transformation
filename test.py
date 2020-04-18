import torch
from metrics import averager

class Tester(object):
    def __init__(self,testloader,criterion,metrics):               
        self.testloader = testloader        
        self.metrics = metrics     
        self.current_batch_nr = 0
        self.total_batch_nr = 0
        self.criterion = criterion
        self.running_loss = averager()
        
    def test_step(self,data, net, use_gpu,on_loss_updated,input_transform=None,*args,**kwargs):
        inputs, labels = data

        if input_transform is not None:
            inputs = input_transform(inputs,*args,**kwargs)  

        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        raw,outputs = net(inputs)

        self.loss = self.criterion(raw, labels)                

        self.running_loss.add(self.loss.item())

        on_loss_updated()
                        
        for metric in self.metrics:
            metric.add(outputs.detach(),labels.detach())

        

    def test_network(self,net,use_gpu,on_loss_updated,input_transform=None,*args,**kwargs):        
        
        self.running_loss.reset()
        self.total_batch_nr = len(self.testloader)

        with torch.no_grad():
            for i,data in enumerate(self.testloader):
                self.current_batch_nr = i
                self.test_step(data, net, use_gpu,on_loss_updated,input_transform,*args,**kwargs)



