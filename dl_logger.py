from torch.utils.tensorboard import SummaryWriter


class dl_logger:    
    
    def log(self,tag,message,idx):
        pass

    def close(self):
        pass

class print_logger(dl_logger):
    
    def log(self,tag,message,idx=-1):
        print("{}: {}".format(tag,message))

class tensorboard_logger(dl_logger):
    
    def __init__(self, current_analysis_name):
        self.writer = SummaryWriter(current_analysis_name)        
        return super().__init__()

    def log(self,tag,message,idx=-1):
        self.writer.add_scalar(tag,message,int(idx))

    def close(self):
        self.writer.close()
