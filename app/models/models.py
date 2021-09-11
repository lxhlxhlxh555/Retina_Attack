import torch
from torchvision import transforms
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import copy
import numpy as np
import torch.nn.functional as F

class RetinaEntropyLoss(nn.Module):
    def __init__(self, class_num=5,device='cpu'):
        super(RetinaEntropyLoss, self).__init__()
        self.class_num = class_num
        self.softmax = nn.Softmax(dim=1)
        self.device = device
    def forward(self, pred, target):
        # print(target)
        prob = F.log_softmax(pred,dim=1)
        target_ = torch.zeros((target.shape[0],self.class_num)).to(self.device)
        target_.scatter_(1, target.view(-1, 1).long(), 1.)
        for i in range(target_.shape[0]):
            if target_[i,0] != 1.0:
                target_[i,1:] = 1.0

        batch_loss = - prob.double() * target_.double()
        batch_loss = batch_loss.sum(dim=1)
        loss = batch_loss.mean()
        return loss

#Definition of PGD attack
class Mask_PGD():
    def __init__(self,model,device):
        super().__init__()
        self.model=model
        self.device = device
    def generate(self,x,**params):
        self.parse_params(**params)
        labels=self.y
        mask = self.mask
 
        adv_x=self.attack(x,labels,mask)
        return adv_x
    def parse_params(self,eps=0.005,iter_eps=0.001,nb_iter=20,clip_min=0.0,clip_max=1,C=0.0,
                     y=None,ord=np.inf,rand_init=False,mask=None):
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
        self.mask = mask
 
    def single_step_attack(self,adv_x,orig_img,labels,mask=None):
        #Get the gradient of x
        adv_x=Variable(adv_x)
        adv_x.requires_grad = True
        # loss_func=nn.CrossEntropyLoss()
        loss_func = RetinaEntropyLoss(device = self.device)
        preds=self.model(adv_x)
        loss = loss_func(preds,labels)
        self.model.zero_grad()
        loss.backward()
        grad=adv_x.grad.data
        # print(mask.shape)
        if mask != None:
            grad = grad * (1-mask)
        #Get the pertubation of a single step
        pertubation=self.iter_eps*torch.sign(grad)
        adv_x = adv_x + pertubation
        pertubation = torch.clamp(adv_x - orig_img,min=-self.eps,max=self.eps)
        adv_img = torch.clamp(orig_img + pertubation,min=self.clip_min,max=self.clip_max)
        return adv_img

    def attack(self,x,labels,mask=None):
        labels = labels.to(self.device)
        # if self.rand_init:
        #     x_tmp=x+torch.Tensor(np.random.uniform(-self.eps, self.eps, x.shape)).type_as(x).cuda()
        # else:
        #     x_tmp=x
        x_tmp = x
        adv_x = copy.deepcopy(x_tmp)
        for i in range(self.nb_iter):
            adv_x = self.single_step_attack(adv_x,x_tmp,labels=labels,mask=mask)
        pertubation_img = adv_x - x
        pertubation_img = pertubation_img.squeeze()
 
        adv_x=torch.clamp(adv_x,self.clip_min,self.clip_max)
 
        return adv_x

#Definition of smooth PGD
class Smooth_PGD():
    def __init__(self,model,device):
        # print("hi")
        super().__init__()
        self.model=model
        self.device = device
    def generate(self,x,**params):
        self.parse_params(**params)
        labels=self.y
        mask = self.mask
 
        adv_x=self.attack(x,labels,mask)
        return adv_x
    def parse_params(self,eps=0.005,iter_eps=0.001,nb_iter=20,clip_min=0.0,clip_max=1,C=0.0,
                     y=None,ord=np.inf,rand_init=False,mask=None,sizes=None,sigmas=None):
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
        self.mask = mask
        self.sizes,self.weights = self.generate_weights(sizes,sigmas)
    def generate_weights(self,sizes,sigmas):
        crafting_sizes = []
        crafting_weights = []
        for size in sizes:
            for sigma in sigmas:
                crafting_sizes.append(size)
                weight = np.arange(size) - size//2
                weight = np.exp(-weight**2.0/2.0/(sigma**2))/np.sum(np.exp(-weight**2.0/2.0/(sigma**2)))
                # print(weight.shape)
                # print(weight.T.shape)
                weight = np.outer(weight,weight)
                # print(weight)
                # print(weight.shape)
                weight = torch.from_numpy(weight).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor).to(self.device)
                # print(weight.shape)
                crafting_weights.append(weight)
        return crafting_sizes, crafting_weights
    def single_step_attack(self,adv_x,orig_img,labels,mask=None):
        #Get the gradient of x
        adv_x=Variable(adv_x)
        adv_x.requires_grad = True
        # loss_func=nn.CrossEntropyLoss()
        loss_func = RetinaEntropyLoss(device=self.device)
        preds=self.model(adv_x)
        loss = loss_func(preds,labels)
        self.model.zero_grad()
        loss.backward()
        grad=adv_x.grad.data
        # print(mask.shape)
        if mask != None:
            grad = grad * (1-mask)
        #Get the pertubation of a single step
        pertubation=self.iter_eps*torch.sign(grad)
        adv_x = adv_x + pertubation
        pertubation = torch.clamp(adv_x - orig_img,min=-self.eps,max=self.eps)
        adv_img = torch.clamp(orig_img + pertubation,min=self.clip_min,max=self.clip_max)
        return adv_img

    def single_step_smooth_attack(self,added,inputs,labels,mask):
        temp = torch.zeros(added.shape).to(self.device)
        temp[:,0,:,:] = F.conv2d(added[:,0,:,:].unsqueeze(0), self.weights[0], padding = self.sizes[0]//2)
        temp[:,1,:,:] = F.conv2d(added[:,1,:,:].unsqueeze(0), self.weights[0], padding = self.sizes[0]//2)
        temp[:,2,:,:] = F.conv2d(added[:,2,:,:].unsqueeze(0), self.weights[0], padding = self.sizes[0]//2)
        for j in range(len(self.sizes)-1):
            temp[:,0,:,:] = temp[:,0,:,:] + F.conv2d(added[:,0,:,:].unsqueeze(0), self.weights[j+1], padding = self.sizes[j+1]//2)
            temp[:,1,:,:] = temp[:,1,:,:] + F.conv2d(added[:,1,:,:].unsqueeze(0), self.weights[j+1], padding = self.sizes[j+1]//2)
            temp[:,2,:,:] = temp[:,2,:,:] + F.conv2d(added[:,2,:,:].unsqueeze(0), self.weights[j+1], padding = self.sizes[j+1]//2)
        temp = temp/float(len(self.sizes))
        output = self.model(inputs + temp)
        # loss_func=nn.CrossEntropyLoss()
        loss_func = RetinaEntropyLoss(device=self.device)
        loss = loss_func(output, labels)
        self.model.zero_grad()
        loss.backward()
        added = added + self.iter_eps * torch.sign(added.grad.data)
        added = torch.clamp(added, -self.eps, self.eps)
        added = torch.autograd.Variable(added.detach().clone(), requires_grad=True)
        return added

    def attack(self,x,labels,mask=None):
        labels = labels.to(self.device)
        x_tmp = x
        adv_x = copy.deepcopy(x_tmp)
        for i in range(self.nb_iter):
            adv_x = self.single_step_attack(adv_x,x_tmp,labels=labels,mask=mask)
        added = adv_x - x_tmp
        added = Variable(added.detach().clone(), requires_grad=True)
        for i in tqdm(range(self.nb_iter*2)):
            added = self.single_step_smooth_attack(added,x_tmp,labels=labels,mask=mask)
        temp = torch.zeros(added.shape).to(self.device)
        temp[:,0,:,:] = F.conv2d(added[:,0,:,:].unsqueeze(0), self.weights[0], padding = self.sizes[0]//2)
        temp[:,1,:,:] = F.conv2d(added[:,1,:,:].unsqueeze(0), self.weights[0], padding = self.sizes[0]//2)
        temp[:,2,:,:] = F.conv2d(added[:,2,:,:].unsqueeze(0), self.weights[0], padding = self.sizes[0]//2)
        # temp = F.conv2d(added, self.weights[0], padding = self.sizes[0]//2)
        for j in range(len(self.sizes)-1):
            # temp = temp + F.conv2d(added, self.weights[j+1], padding = self.sizes[j+1]//2)
            temp[:,0,:,:] = temp[:,0,:,:] + F.conv2d(added[:,0,:,:].unsqueeze(0), self.weights[j+1], padding = self.sizes[j+1]//2)
            temp[:,1,:,:] = temp[:,1,:,:] + F.conv2d(added[:,1,:,:].unsqueeze(0), self.weights[j+1], padding = self.sizes[j+1]//2)
            temp[:,2,:,:] = temp[:,2,:,:] + F.conv2d(added[:,2,:,:].unsqueeze(0), self.weights[j+1], padding = self.sizes[j+1]//2)
        temp = temp/float(len(self.sizes))
        temp = torch.clamp(temp, -self.eps, self.eps)
        crafting_output = x_tmp + temp.detach()
        adv_x = crafting_output.clone()

        transform = transforms.ToPILImage()
        pertubation_img = adv_x - x
        pertubation_img = transform(pertubation_img.squeeze())
        # r, g, b = pertubation_img.getpixel((112, 112))
        # pertubation_img.save('pertubation.jpeg')
        adv_x=adv_x.cpu().detach().numpy()
 
        adv_x=np.clip(adv_x,self.clip_min,self.clip_max)
 
        return adv_x

