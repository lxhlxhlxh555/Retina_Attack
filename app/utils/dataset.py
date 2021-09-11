import os
import skimage
import numpy as np
from torch.utils.data import Dataset, dataset
import json
import cv2
from skimage import io
from torchvision.transforms import transforms
from .attack_tools import pad_img
class LocalDataset(Dataset):
    def __init__(self, root_dir, resize=448):
        self.root_dir = root_dir
        self.imgs_dir = os.path.join(self.root_dir,'images')
        self.label_dir = os.path.join(self.root_dir,'labels.json')
        self.imgs = os.listdir(self.imgs_dir)
        self.resize = resize
        with open(self.label_dir) as f:
            self.labels = json.load(f)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_name = self.imgs[index]
        img_dir = os.path.join(self.imgs_dir,img_name)
        img = cv2.imread(img_dir)
        # print(index)
        # print(img_name)
        # print(img.shape)
        img = pad_img(img)
        img = cv2.resize(img,(self.resize,self.resize))
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        # print(img.shape)
        t = transforms.ToTensor()
        img = t(img.astype(np.uint8))
        label = int(self.labels[img_name])
        sample = {'image':img, 'label':label}
        return sample
    
class GeneratedDataset(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        return {'image':self.imgs[index],'label':self.labels[index]}

def build_dataset(params):
    dataset_type = params['dataset']
    if dataset_type == 'local':
        assert('img_dir' in params)
        dataset = LocalDataset(params['img_dir'])
    return dataset
