import torch

class averager(object):

    def reset(self):
        self.counter = 0
        self.running_value = 0.0        
    
    def __init__(self):
        self.reset()        
        return super().__init__()

    def add(self,value):
        self.running_value += value
        self.counter += 1

    def get_average(self):
        if self.counter < 1:
            return 0
        else:
            return self.running_value/self.counter

#class metric:
#    def add(self,outputs,labels):
#        """ Add batch to metric """
#        pass

#    def get_latest(self):
#        """ Get the latest values added """
#        pass

#    def get_average(self):
#        """ Get the average of all values """
#        pass

#    def reset(self):
#        """ Reset the metric """
#        pass

#class accuracy(metric):    

#    def __init__(self, name,*args, **kwargs):
#        self._averager = averager()
#        self._latest = 0.0 
#        self.name = name
#        return super().__init__(*args, **kwargs)

#    def add(self,outputs,labels):
#        _, predicted = torch.max(outputs.data, 1)                
#        #correct = (predicted == labels).sum().item()
#        #total = len(predicted)

#        _, predicted = torch.max(outputs.data, 1)
#        correct_tmp = (predicted == labels)
#        correct = correct_tmp.sum().item()/correct_tmp.nelement()
#        self._latest = correct
#        self._averager.add(correct)

#    def get_latest(self):
#        return self._latest

#    def get_average(self):
#        return self._averager.get_average()

#    def reset(self):
#        self._latest = 0.0
#        self._averager.reset()

#class criterion(metric):    

#    def __init__(self, criterion,name,*args, **kwargs):
#        self._averager = averager()
#        self._latest = 0.0 
#        self.name = name
#        self.criterion = criterion
#        return super().__init__(*args, **kwargs)

#    def add(self,outputs,labels):
#        loss = self.criterion(outputs, labels)
#        #loss.backward()
#        self._latest = loss.item()
#        self._averager.add(loss.item())

#    def get_latest(self):
#        return self._latest

#    def get_average(self):
#        return self._averager.get_average()

#    def reset(self):
#        self._latest = 0.0
#        self._averager.reset()


import torch
import numpy as np
from sklearn import metrics as skmet

class ConfMatrix(object):
    

    def __init__(self, num_classes, ignore_index=None):
        super().__init__()
        #self.conf_metric = ConfusionMatrix(num_classes, normalized)
        self.conf_metric = np.zeros([num_classes,num_classes])
        self.num_classes = num_classes

        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def reset(self):
        #self.conf_metric.reset()
        self.conf_metric = np.zeros([self.num_classes,self.num_classes])

    def add(self, predicted, target):

        self.shape = predicted.shape.copy()

        _, predicted = predicted.max(1)

        self.conf_metric += skmet.confusion_matrix(target.view(-1).cpu().numpy(),predicted.view(-1).cpu().numpy(),np.arange(0,self.num_classes)) 

        

        #self.conf_metric.add(predicted.view(-1), target.view(-1))

    def get_mean_metrics(self):

        print("There are {} Test examples in the Confusion Matrix (This should be an even number)".format(self.conf_metric.sum()/self.shape))

        conf_matrix = self.conf_metric.copy()

        if self.ignore_index is not None:
            for index in self.ignore_index:
                conf_matrix[:, self.ignore_index] = 0
                conf_matrix[self.ignore_index, :] = 0
        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, 0) - true_positive
        false_negative = np.sum(conf_matrix, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)

            accuracy = true_positive  / np.sum(conf_matrix)

            global_accuracy = true_positive.sum()/ np.sum(conf_matrix)

            precision = true_positive / (true_positive + false_positive)

            recall = true_positive / (true_positive + false_negative)

        out_dict = {"mIOU":np.nanmean(iou),                    
                    "MeanAccuracy":np.nanmean(accuracy),
                    "GlobalAccuracy":global_accuracy,
                    "Precision":np.nanmean(precision),
                    "Recall":np.nanmean(recall)}

        return out_dict

    def get_per_class_accuracy(self):
        conf_matrix = self.conf_metric.copy()

        if self.ignore_index is not None:
            for index in self.ignore_index:
                conf_matrix[:, self.ignore_index] = 0
                conf_matrix[self.ignore_index, :] = 0
        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, 0) - true_positive
        false_negative = np.sum(conf_matrix, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):            
            accuracy = true_positive  / conf_matrix.sum(axis=0)

        out_dict = {}
        for i,val in enumerate(accuracy):
            out_dict[str(i)] = val

        return out_dict
    #def value(self):
    #    """Computes the IoU and mean IoU.

    #    The mean computation ignores NaN elements of the IoU array.

    #    Returns:
    #        Tuple: (IoU, mIoU). The first output is the per class IoU,
    #        for K classes it's numpy.ndarray with K elements. The second output,
    #        is the mean IoU.
    #    """
    #    #conf_matrix = self.conf_metric.value()
    #    conf_matrix = self.conf_metric

    #    if self.ignore_index is not None:
    #        for index in self.ignore_index:
    #            conf_matrix[:, self.ignore_index] = 0
    #            conf_matrix[self.ignore_index, :] = 0
    #    true_positive = np.diag(conf_matrix)
    #    false_positive = np.sum(conf_matrix, 0) - true_positive
    #    false_negative = np.sum(conf_matrix, 1) - true_positive

    #    # Just in case we get a division by 0, ignore/hide the error
    #    with np.errstate(divide='ignore', invalid='ignore'):
    #        iou = true_positive / (true_positive + false_positive + false_negative)

    #        accuracy = true_positive  / np.sum(conf_matrix)

    #        precision = true_positive / (true_positive + false_positive)

    #        recall = true_positive / (true_positive + false_negative)

    #    return iou, np.nanmean(iou)