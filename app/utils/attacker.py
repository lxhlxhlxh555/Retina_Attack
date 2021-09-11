import abc
import albumentations
import numpy as np
from .attack_tools import add_gaussian_noise,add_sp_noise
import cv2
from .dataset import GeneratedDataset
import torchvision.transforms as transforms
import torch
from ..models.models import Mask_PGD
class Attacker(object):
    def __init__(self):
        pass
    @abc.abstractmethod
    def run(self):
        pass

class NormalAttacker(Attacker):
    def __init__(self,params):
        self.attack_types = params['attack_types']
        self.attack_levels = params['attack_levels']
    def run(self,model,dataset,device):
        attack_dict = {}
        attack_list = []
        assert(len(self.attack_types)==len(self.attack_levels))
        for i,type in enumerate(self.attack_types):
            attack_dict[type] = self.attack_levels[i]
        if 'defocus_blur' in self.attack_types:
            level_list = [1,3,5,7,9]
            level = attack_dict['defocus_blur']
            blur_limit = level_list[int(level)-1]
            attack_list.append(albumentations.GaussianBlur(blur_limit=blur_limit, p=1))
        if 'motion_blur' in self.attack_types:
            level_list = [10,30,50,70,90]
            level = attack_dict['motion_blur']
            blur_limit = level_list[int(level)-1]
            attack_list.append(albumentations.MotionBlur(blur_limit=blur_limit,p=1))
        if 'rgb_shift' in self.attack_types:
            attack_list.append(albumentations.RGBShift(p=1))
        if 'hsv_shift' in self.attack_types:
            level_list = [5,10,15,20,25]
            level = attack_dict['hsv_shift']
            shift_limit = level_list[int(level)-1]
            attack_list.append(albumentations.HueSaturationValue(hue_shift_limit=shift_limit, sat_shift_limit=shift_limit, val_shift_limit=shift_limit, p=1))
        if 'brightness_contrast' in self.attack_types:
            level_list = [0.1,0.2,0.3,0.4,0.5]
            level = attack_dict['brightness_contrast']
            limit = level_list[int(level)-1]
            attack_list.append(albumentations.RandomBrightnessContrast(brightness_limit=limit, contrast_limit=limit, p=1))
        album = albumentations.Compose(attack_list)
        adv_imgs = []
        labels = []
        for item in dataset:
            transform = transforms.ToTensor()
            reverse = transforms.ToPILImage()
            img = item['image']
            label = item['label']
            img = reverse(img.squeeze())
            img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
            adv_img = album(image=img)["image"]
            if 'iso_noise' in self.attack_types:
                mean_list = [2,5,10,15,20]
                sigma_list = [30,40,50,60,70]
                level = attack_dict['iso_noise']
                idx = int(level) - 1
                adv_img = add_gaussian_noise(adv_img,[mean_list[idx],sigma_list[idx]],p=1)
            if 'sp_noise' in self.attack_types:
                level_list = [0.9,0.8,0.7,0.6,0.5]
                level = attack_dict['sp_noise']
                snr = level_list[int(level)-1]
                adv_img = add_sp_noise(adv_img,SNR=snr,p=1)
            adv_img = cv2.cvtColor(adv_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            adv_img = transform(adv_img.astype(np.uint8))
            adv_imgs.append(adv_img)
            labels.append(label)
        new_dataset = GeneratedDataset(adv_imgs, labels)
        return new_dataset


class PGDAttacker(Attacker):
    def __init__(self,params):
        self.level = params['adv_level']
        self.set_attack_level()
    def set_attack_level(self):
        if self.level == 1:
            print('Level:1')
            self.eps=0.005
            self.iter_eps=0.001
            self.nb_iter=20
        elif self.level == 2:
            print('Level:2')
            self.eps=0.02
            self.iter_eps = 0.002
            self.nb_iter = 40
        elif self.level == 3:
            print('Level:3')
            self.eps = 0.1
            self.iter_eps = 0.003
            self.nb_iter = 60
    def run(self,model,dataset,device):
        model = model.to(device)
        attacker = Mask_PGD(model,device) 
        adv_imgs = []
        labels = [] 
        for item in dataset:
            img = item['image']
            label = item['label']
            h,w,_ = img.shape
            img = img.unsqueeze(0)
            print("Start Attacking...")
            adv_img = attacker.generate(img.to(device),y=torch.tensor([label]).to(device),eps=self.eps,iter_eps=self.iter_eps,nb_iter=self.nb_iter,mask=None)
            # adv_img = attacker.generate(img.to(device),y=torch.tensor([labels[count]]).to(device),eps=eps,iter_eps=iter_eps,nb_iter=nb_iter,mask=None,sizes=sizes,sigmas=sigmas)
            adv_img = adv_img.squeeze(0)
            print(adv_img.shape)
            adv_imgs.append(adv_img)
            labels.append(label)
        new_dataset = GeneratedDataset(adv_imgs, labels)
        return new_dataset
           
def build_attacker(params):
    attack_type = params['type']
    if attack_type == 'normal':
        attacker = NormalAttacker(params)
    elif attack_type == 'pgd':
        attacker = PGDAttacker(params)
    return attacker