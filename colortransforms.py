import torch
import numpy as np



def madden(image,alpha):    
    '''Madden Color transform recreated after http://www.robots.ox.ac.uk/~mobile/Papers/2014ICRA_maddern.pdf'''
    eps=1e-7
    
    madden = 0.5 + torch.log(image[:,1,:,:]+eps) - alpha * torch.log(image[:,2,:,:]+eps) - (1-alpha) * torch.log(image[:,0,:,:]+eps)
   
    return madden.view([image.shape[0],1,image.shape[2],image.shape[3]])

def madden_nan_rm(image,alpha):    
    
    
    madden = 0.5 + torch.log(image[:,1,:,:]) - alpha * torch.log(image[:,2,:,:]) - (1-alpha) * torch.log(image[:,0,:,:])
    
    #Set all infs to zero maximum value
    madden[madden == float("Inf")] = np.nanmax(madden[madden != np.inf]).item()

    return madden.view([image.shape[0],1,image.shape[2],image.shape[3]])

#
def hsv(im):
        ''' HSV Transform taken from: https://github.com/odegeasslbc/Differentiable-RGB-to-HSV-convertion-pytorch/blob/master/pytorch_hsv.py'''
        img = im * 0.5 + 0.5
        hue = torch.Tensor(im.shape[0], im.shape[2], im.shape[3]).to(im.device)
        eps=1e-7
        hue[ img[:,2]==img.max(1)[0] ] = 4.0 + ( (img[:,0]-img[:,1]) / ( img.max(1)[0] - img.min(1)[0] + eps) ) [ img[:,2]==img.max(1)[0] ]
        hue[ img[:,1]==img.max(1)[0] ] = 2.0 + ( (img[:,2]-img[:,0]) / ( img.max(1)[0] - img.min(1)[0] + eps) ) [ img[:,1]==img.max(1)[0] ]
        hue[ img[:,0]==img.max(1)[0] ] = (0.0 + ( (img[:,1]-img[:,2]) / ( img.max(1)[0] - img.min(1)[0] + eps) ) [ img[:,0]==img.max(1)[0] ]) % 6

        hue[img.min(1)[0]==img.max(1)[0]] = 0.0
        hue = hue/6

        saturation = ( img.max(1)[0] - img.min(1)[0] ) / ( img.max(1)[0] + eps )
        saturation[ img.max(1)[0]==0 ] = 0

        value = img.max(1)[0]
        return hue, saturation, value




def madden_hs(image,alpha):           
    madden_out = madden(image,alpha)
    hue, saturation, value = hsv(image)       
    image[:,0,:,:] = madden_out.squeeze(1)
    image[:,1,:,:] = hue
    image[:,2,:,:] = saturation    
    return image