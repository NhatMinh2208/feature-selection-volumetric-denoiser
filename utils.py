import mitsuba as mi
import numpy as np
import torch 
def read_exr(src) -> mi.Bitmap: 
    bmp_exr = mi.Bitmap(src)
    return bmp_exr

def bitmap2array(bmp : mi.Bitmap) -> dict:
    data = dict(bmp.split())
    for k, v in data.items():
        data[k] = np.array(v)
        #expand array dimension if not 3
        if(data[k].ndim == 2):
            data[k] = np.expand_dims(data[k], axis=2)
    return data



def crop(img, patch_size = 64):
    img_width, img_height, _ = img['a'].shape
    patches = []
    for i in range(0, img_height, patch_size):
        for j in range(0, img_width, patch_size):
            patch = {key: val[j : (j+patch_size), i : (i+patch_size), : ] for key, val in img.items()}
            patches.append(patch)
    return patches

def merge(ref_img, patches, patch_size = 64):
    img = {}
    img_width, img_height, _ = ref_img['a'].shape
    for key, val in patches[0].items():
        _, _, img_channels = patches[0][key].shape
        img[key] = np.zeros((img_width, img_height, img_channels))
    step = 0
    for i in range(0, img_height, patch_size):
        for j in range(0, img_width, patch_size):
            for key, val in patches[step].items():
                img[key][j : (j+patch_size), i : (i+patch_size), : ] =  val
            step += 1
    return img


def log_transform(x):
    return torch.log(torch.abs(x) + 1)

def mean(dataloader):
    #mean = torch.zeros(3)
    mean = torch.zeros(1)
    #mean_squared_image = torch.zeros(3)
    for data in dataloader:
        #mean += torch.mean(data['z_v_ss'], dim=(1, 2))
        #mean_squared_image += torch.mean(data ** 2, dim=(1, 2))
        mean += torch.mean(data['z_v_ss'])
    return mean / len(dataloader)#, mean_squared_image

def mean_standardDeviation(dataloader):
    #https://stackoverflow.com/questions/73350133/how-to-calculate-mean-and-standard-deviation-of-a-set-of-images
    # mu, mu_squared_image = mean(dataloader)
    # variance = mu_squared_image - mu ** 2
    mu = mean(dataloader)
    #variance = torch.zeros(3)
    variance = torch.zeros(1)
    for data in dataloader:
        #variance += torch.mean((data['z_v_ss'] - mu)**2, dim=(1, 2))
        variance += torch.mean((data['z_v_ss'] - mu)**2)
    stddev = torch.sqrt(variance / len(dataloader) )
    return mu, stddev

def normal_transform(x, mu, sigma):
    #mu, sigma = mean_standardDeviation(x)
    return (x - mu) / sigma

def sign_log_transform(x):
    mu, sigma = mean_standardDeviation(x)
    return torch.sign(x -mu) * torch.log((torch.abs(x - mu) / sigma) + 1)

