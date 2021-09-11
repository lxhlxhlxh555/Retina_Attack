import numpy as np
import random
import torch
#Pad an image to make it square
def pad_img(img):
    # print(len(img.shape))
    if len(img.shape) == 3:
        x,y,c = img.shape
    else:
        x,y = img.shape
        c = 3
        img = img[:,:,np.newaxis]
        img = np.repeat(img, c, axis=2)

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

def load_model(model_dir,device):
    if device == 'cpu':
        model = torch.jit.load(model_dir)
    else:
        model = torch.jit.load(model_dir)
    model = model.to(device)
    return model