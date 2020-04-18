class visualizer(object):
    '''Class to visualize images stored in tensors'''

    def image_tensor_to_image(self,img):
    
        to_pil = torchvision.transforms.ToPILImage()
        pil_img = to_pil(img)

        return pil_img   

    def label_tensor_to_image(self,lab,class_encoding):

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
