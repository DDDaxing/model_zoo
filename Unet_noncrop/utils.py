from os.path import join
from os import listdir
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np

class LocalDataset(Dataset):
    """Local dataset."""

    def __init__(self, img_dir, target_dir, transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_dir = img_dir
        self.target_dir = target_dir
        self.transform = transform
        self.img_name = [join(self.img_dir, x) for x in listdir(img_dir)]
        self.target_name = [join(self.target_dir, x) for x in listdir(target_dir)]


    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, idx):
        img = Image.open(self.img_name[idx])
        target = Image.open(self.target_name[idx])
        
        if self.transform:
            img = self.transform(img)
            target = self.transform(target)

        return img, target


def im_show(img_list,in_idx):

    to_PIL = transforms.ToPILImage()
    if len(img_list) > 4:
        raise Exception("len(img_list) must be smaller than 5")

    for idx, img in enumerate(img_list):
        
        img = img[1,:,:,:]
        img = img.cpu().detach().numpy()
        #img = np.array(to_PIL(img))
        img = to_PIL(torch.Tensor(img))
        plt.subplot(121+idx)
        fig = plt.imshow(img, cmap='gray')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    plt.show()
    plt.savefig('targ_pred_img'+ str(in_idx) + '.png')


def crop_tensor(x, size):
    # now assume the image is square
    img_size = x.size()[2]
    r = (img_size - size)//2
    w = img_size - r - (img_size - size)%2
    return x[:,:, r:w, r:w]

