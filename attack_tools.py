import torch
import numpy as np
import random
from torch.autograd import Variable
import torch.nn as nn
from torchvision import transforms
import copy

#Pad an image to make it square
def pad_img(img):
    x,y,c = img.shape
    if x > y:
        pad_len = int((x-y)/2)
        padded = np.zeros((x,x,3))
        if (x-y) % 2 == 0:
            padded[:,pad_len:x-pad_len,:] = img
        else:
            padded[:,pad_len+1:x-pad_len,:] = img
    else:
        pad_len = int((y-x)/2)
        padded = np.zeros((y,y,3))
        if (y-x) % 2 == 0:
            padded[pad_len:y-pad_len,:,:] = img
        else:
            padded[pad_len+1:y-pad_len,:,:] = img
    return padded

#Add gaussian noise to an image
def add_gaussian_noise(pic,noise_sigma=[10,50],p=0.3):
    num = np.random.rand()
    if num >= p:
        return pic
    temp_image = np.float64(np.copy(pic))
    sigma_range = np.arange(noise_sigma[0],noise_sigma[1])
    sigma = random.sample(list(sigma_range),1)[0]
    h, w, _ = temp_image.shape
    #Standard norm * noise_sigma
    noise = np.random.randn(h, w) * sigma
 
    noisy_image = np.zeros(temp_image.shape, np.float64)
    if len(temp_image.shape) == 2:
        noisy_image = temp_image + noise
    else:
        noisy_image[:,:,0] = temp_image[:,:,0] + noise
        noisy_image[:,:,1] = temp_image[:,:,1] + noise
        noisy_image[:,:,2] = temp_image[:,:,2] + noise
        
    return noisy_image

#Add salt and pepper noise to an image
def add_sp_noise(pic,SNR=0.9,p=0.3):
    num = np.random.rand()
    if num >= p:
        return pic
    noisy_image = pic.copy()
    h, w, c = noisy_image.shape
    mask = np.random.choice((0, 1, 2), size=(1, h, w), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
    mask = np.repeat(mask, c, axis=0)     # copy the mask by channel
    mask = np.transpose(mask,(1,2,0))
    noisy_image[mask == 1] = 255    # Salt
    noisy_image[mask == 2] = 0      # Pepper
    return noisy_image

#Definition of PGD attack
class PGD():
    def __init__(self,model,device):
        super().__init__()
        self.model=model
        self.device = device
    def generate(self,x,**params):
        self.parse_params(**params)
        labels=self.y
 
        adv_x=self.attack(x,labels)
        return adv_x
    def parse_params(self,eps=0.005,iter_eps=0.001,nb_iter=20,clip_min=0.0,clip_max=1,C=0.0,
                     y=None,ord=np.inf,rand_init=False):
        self.eps=eps
        self.iter_eps=iter_eps
        self.nb_iter=nb_iter
        self.clip_min=clip_min
        self.clip_max=clip_max
        self.y=y
        self.ord=ord
        self.rand_init=rand_init
        self.model.to(self.device)
        self.C=C
 
    def sigle_step_attack(self,adv_x,orig_img,labels):
        #Get the gradient of x
        adv_x=Variable(adv_x)
        adv_x.requires_grad = True
        loss_func=nn.CrossEntropyLoss()
        preds=self.model(adv_x)
        loss = loss_func(preds,labels)
        self.model.zero_grad()
        loss.backward()
        grad=adv_x.grad.data

        #Get the pertubation of a single step
        pertubation=self.iter_eps*torch.sign(grad)
        adv_x = adv_x + pertubation
        pertubation = torch.clamp(adv_x - orig_img,min=-self.eps,max=self.eps)
        adv_img = torch.clamp(orig_img + pertubation,min=self.clip_min,max=self.clip_max)
        return adv_img

    def attack(self,x,labels):
        labels = labels.to(self.device)
        print(self.rand_init)
        # if self.rand_init:
        #     x_tmp=x+torch.Tensor(np.random.uniform(-self.eps, self.eps, x.shape)).type_as(x).cuda()
        # else:
        #     x_tmp=x
        x_tmp = x
        adv_x = copy.deepcopy(x_tmp)
        for i in range(self.nb_iter):
            adv_x = self.sigle_step_attack(adv_x,x_tmp,labels=labels)
        transform = transforms.ToPILImage()
        pertubation_img = adv_x - x
        pertubation_img = transform(pertubation_img.squeeze())
        r, g, b = pertubation_img.getpixel((112, 112))
        print(r,g,b)
        pertubation_img.save('pertubation.jpeg')
        adv_x=adv_x.cpu().detach().numpy()
 
        adv_x=np.clip(adv_x,self.clip_min,self.clip_max)
 
        return adv_x

