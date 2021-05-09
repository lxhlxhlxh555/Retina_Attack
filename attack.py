import os
import cv2
import torch
import argparse
import numpy as np
import albumentations
import torch.nn.functional as F
from tqdm import tqdm
from attack_tools import pad_img,add_gaussian_noise,add_sp_noise,PGD
from torchvision import transforms

#Nomal attack
def normal_attack(args):
    img = cv2.imread(args.pic_dir)
    img = pad_img(img)
    attack_list = []
    if args.gaussian_blur:
        attack_list.append(albumentations.GaussianBlur(blur_limit=5, p=1))
    if args.motion_blur:
        attack_list.append(albumentations.MotionBlur(blur_limit=50,p=1))
    if args.rgb_shift:
        attack_list.append(albumentations.RGBShift(p=1))
    if args.hsv_shift:
        attack_list.append(albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=1))
    if args.brightness_contrast:
        attack_list.append(albumentations.OneOf([
                albumentations.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                albumentations.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1)
            ],p=1))
    album = albumentations.Compose(attack_list)
    img = img.astype(np.uint8)
    adv_img = album(image=img)["image"]
    if args.gaussian_noise:
        adv_img = add_gaussian_noise(adv_img,p=1)
    if args.sp_noise:
        adv_img = add_sp_noise(adv_img,p=1)
    #Save img to user defined direction
    cv2.imwrite(args.origin_dir,img)
    cv2.imwrite(args.output_dir,adv_img)
    return adv_img


#Adversarial attack
def adversarial_attack(args):
    #load images and resize
    img = cv2.imread(args.pic_dir)
    img = pad_img(img)
    h,w,c = img.shape
    print(h,w)
    img = cv2.resize(img,(448,448))
    transform = transforms.ToTensor()
    reverse = transforms.ToPILImage()
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    a = reverse(img.astype(np.uint8))
    a = a.resize((h,w))
    a.save(args.origin_dir)
    img = transform(img.astype(np.uint8))

    #Define model and attacker
    label = args.ground_truth
    model = torch.jit.load(args.model_dir)
    model = model.to(args.device)
    attacker = PGD(model,args.device)

    #Prepare the level of attack
    level = args.adv_level
    if level == 1:
        eps=0.005
        iter_eps=0.001
        nb_iter=20
    elif level == 2:
        eps=0.01
        iter_eps = 0.002
        nb_iter = 40
    elif level == 3:
        eps = 0.1
        iter_eps = 0.003
        nb_iter = 60

    #Generate adversarial image
    img = img.unsqueeze(0)
    old_logits = model(img.to(args.device))
    old_probas = np.array(F.softmax(old_logits, dim=-1).cpu().detach())
    y = torch.max(old_logits, 1)[1].cpu().numpy()[0]
    adv_img = attacker.generate(img.to(args.device),y=torch.tensor([0]).to(args.device),eps=eps,iter_eps=iter_eps,nb_iter=nb_iter)
    print(adv_img.shape)
    adv_img = torch.tensor(adv_img).to(args.device)
    logits = model(adv_img)
    probas = np.array(F.softmax(logits, dim=-1).cpu().detach())
    new_y = torch.max(logits, 1)[1].cpu().numpy()[0]

    #Save adversarial image
    img = reverse(adv_img.squeeze())
    img = img.resize((h,w))
    img.save(args.output_dir)

    #Print the results
    print("Original ligits:{};New logits:{}".format(old_logits,logits))
    print("Original probas:{};New probas:{}".format(old_probas,probas))
    print("Original output:{}; New output:{}".format(y,new_y))
    print("Ground Truth:%d" %(label))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #General settings
    parser.add_argument('--pic_dir',type=str,default='./799_right.jpeg',help="Choose to do adversarial augmentation")
    parser.add_argument('--origin_dir',type=str,default='./origin.jpeg',help='The direction of origin image')
    parser.add_argument('--output_dir',type=str,default='./adv.jpeg',help='The direction of output image')
    parser.add_argument('--attack_type',type=str,default='adversarial',choices=['adversarial','normal'],help="Choose an attack type")
    #Adversarial attack arguments
    parser.add_argument('--device',type=str,default='cpu',help='Define model device')
    parser.add_argument('--model_dir',type=str,default='./jit_module_cpu.pth',help="The direction of model")
    parser.add_argument('--ground_truth',type=int,default=0,choices=[0,1,2,3,4],help="The ground truth of the picture")
    parser.add_argument('--adv_level',type=int,default=1,choices=[1,2,3],help="Choose an attack level")
    #Normal attack arguments
    parser.add_argument('-gn','--gaussian_noise',default=False,action='store_true',help="Add gaussian noise")
    parser.add_argument('-gb','--gaussian_blur',default=False,action='store_true',help="Add gaussian blur")
    parser.add_argument('-sp','--sp_noise',default=False,action='store_true',help="Add salt and pepper noise")
    parser.add_argument('-mb','--motion_blur',default=False,action='store_true',help="Add motion blur")
    parser.add_argument('-rgb','--rgb_shift',default=False,action='store_true',help="Add rgb shift")
    parser.add_argument('-hsv','--hsv_shift',default=False,action='store_true',help="Add hsv shift")
    parser.add_argument('-bc','--brightness_contrast',default=False,action='store_true',help="Add brightness contrast change")

    args = parser.parse_args()
    if args.attack_type == 'adversarial':
        adversarial_attack(args)
    else:
        normal_attack(args)




