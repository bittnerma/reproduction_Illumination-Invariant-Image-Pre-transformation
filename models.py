import torch
import torch.nn as nn
import torch.nn.functional as F

#Network model used on MNIST
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)        
                
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))        
        x = F.relu(self.fc2(x))        
        x = self.fc3(x)
        return x

import torchvision.models as models
vgg16 = models.vgg16(pretrained=True)

#This breaks currently 
use_gpu = torch.cuda.is_available()

#Network model used on MNIST
class OwnSegNet(nn.Module):
    def __init__(self,input_size):
        super(OwnSegNet, self).__init__()
        self.enc_l1_1 = nn.Conv2d(input_size, 64, 3,padding=1)
        self.enc_l1_2 = nn.Conv2d(64, 64, 3,padding=1)
        self.pool = nn.MaxPool2d(2, 2,return_indices=True)
        self.enc_l2_1 = nn.Conv2d(64, 128, 3,padding=1)
        self.enc_l2_2 = nn.Conv2d(128, 128, 3,padding=1)

        self.enc_l3_1 = nn.Conv2d(128, 256, 3,padding=1)
        self.enc_l3_2 = nn.Conv2d(256, 256, 3,padding=1)
        self.enc_l3_3 = nn.Conv2d(256, 256, 3,padding=1)

        self.enc_l4_1 = nn.Conv2d(256, 512, 3,padding=1)
        self.enc_l4_2 = nn.Conv2d(512, 512, 3,padding=1)
        self.enc_l4_3 = nn.Conv2d(512, 512, 3,padding=1)

        self.enc_l5_1 = nn.Conv2d(512, 512, 3,padding=1)
        self.enc_l5_2 = nn.Conv2d(512, 512, 3,padding=1)
        self.enc_l5_3 = nn.Conv2d(512, 512, 3,padding=1)
        
        self.unpool = nn.MaxUnpool2d(2,2)

        self.dec_l5_1 = nn.Conv2d(512, 512, 3,padding=1)
        self.dec_l5_2 = nn.Conv2d(512, 512, 3,padding=1)
        self.dec_l5_3 = nn.Conv2d(512, 512, 3,padding=1)

        self.dec_l4_1 = nn.Conv2d(512, 512, 3,padding=1)
        self.dec_l4_2 = nn.Conv2d(512, 512, 3,padding=1)
        self.dec_l4_3 = nn.Conv2d(512, 256, 3,padding=1)

        self.dec_l3_1 = nn.Conv2d(256, 256, 3,padding=1)
        self.dec_l3_2 = nn.Conv2d(256, 256, 3,padding=1)
        self.dec_l3_3 = nn.Conv2d(256, 128, 3,padding=1)

        self.dec_l2_1 = nn.Conv2d(128, 128, 3,padding=1)
        self.dec_l2_2 = nn.Conv2d(128, 64, 3,padding=1)

        self.dec_l1_1 = nn.Conv2d(64, 64, 3,padding=1)
        self.dec_l1_2 = nn.Conv2d(64, 12, 3,padding=1)
        
        #Initialize the weights

        vgg_like_layers = [self.enc_l1_1,
        self.enc_l1_2,
        self.enc_l2_1,
        self.enc_l2_2,
        self.enc_l3_1,
        self.enc_l3_2,
        self.enc_l3_3,
        self.enc_l4_1,
        self.enc_l4_2,
        self.enc_l4_3,
        self.enc_l5_1,
        self.enc_l5_2,
        self.enc_l5_3]

        i = 0
        
        for feature in vgg16.features:
            if isinstance(feature,nn.Conv2d):
                if vgg_like_layers[i].weight.shape == feature.weight.shape and vgg_like_layers[i].bias.shape == feature.bias.shape:
                    vgg_like_layers[i].weight = feature.weight
                    vgg_like_layers[i].bias = feature.bias
                    i += 1
                    print("Initialized layer {}".format(i))
                    if i >= len(vgg_like_layers):
                        break
        
        

        
        
                
    def forward(self, x):
        dim1 = x.shape
        x = F.relu(self.enc_l1_1(x))
        x = F.relu(self.enc_l1_2(x))
        x,idx_l1 = self.pool(x)
        
        dim2 = x.shape

        x = F.relu(self.enc_l2_1(x))
        x = F.relu(self.enc_l2_2(x))
        x,idx_l2 = self.pool(x)

        dim3 = x.shape

        x = F.relu(self.enc_l3_1(x))
        x = F.relu(self.enc_l3_2(x))
        x = F.relu(self.enc_l3_3(x))
        x,idx_l3 = self.pool(x)

        dim4 = x.shape

        x = F.relu(self.enc_l4_1(x))
        x = F.relu(self.enc_l4_2(x))
        x = F.relu(self.enc_l4_3(x))
        x,idx_l4 = self.pool(x)

        dim5 = x.shape

        x = F.relu(self.enc_l5_1(x))
        x = F.relu(self.enc_l5_2(x))
        x = F.relu(self.enc_l5_3(x))
        x,idx_l5 = self.pool(x)

        x = self.unpool(x,idx_l5,output_size = dim5)
        x = F.relu(self.dec_l5_1(x))
        x = F.relu(self.dec_l5_2(x))
        x = F.relu(self.dec_l5_3(x))

        x = self.unpool(x,idx_l4,output_size = dim4)
        x = F.relu(self.dec_l4_1(x))
        x = F.relu(self.dec_l4_2(x))
        x = F.relu(self.dec_l4_3(x))

        x = self.unpool(x,idx_l3,output_size = dim3)
        x = F.relu(self.dec_l3_1(x))
        x = F.relu(self.dec_l3_2(x))
        x = F.relu(self.dec_l3_3(x))

        x = self.unpool(x,idx_l2,output_size = dim2)
        x = F.relu(self.dec_l2_1(x))
        x = F.relu(self.dec_l2_2(x))

        x = self.unpool(x,idx_l1,output_size = dim1)
        x = F.relu(self.dec_l1_1(x))
        x = F.relu(self.dec_l1_2(x))

        x = F.softmax(x, dim=1)       

        return x
