# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 11:58:15 2022

@author: lbnc
"""
from pathlib import Path
import torch
#=======================
# Download pretrained Inception network weights and biases model = torch.hub.load("pytorch/vision", "inception_v3",
weights="IMAGENET1K_V1"
#=======================
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms 
import torchvision.datasets as datasets

# before each image is pushed through the network, it is transformed class Transform:
class AuroraClassifier:
    def __init__(self):
        self.transform = transforms.Compose([
        # by randomly rotating
        transforms.RandomRotation(90),
        # by cropping the center rectangle of size 132 px
        transforms.CenterCrop(132),
        # then cropping again a 64x64 part of the image
        transforms.RandomCrop(64),
        # resizing it to match the input size needed for Inception
        transforms.Resize(299,
            interpolation=transforms.InterpolationMode.BICUBIC),
        # maybe flip the image
        transforms.RandomHorizontalFlip(p=0.5),
        # and convert to pytorch tensor
        transforms.ToTensor()
        ])
        # a little helper routine
        self.to_tensor = transforms.Compose([
        transforms.ToTensor()
        ])

    # this is called as an image (in variable x) is loaded
    def __call__(self, x):
        _x = self.to_tensor(x)
        # find mean and stdev for normalization
        mn = torch.mean(_x[1,:,:])
        sd = torch.std(_x[1,:,:])
        # transform image (see above)
        x = self.transform(x)
        # normalize
        x[0,:,:] = (x[1,:,:] - mn)/sd
        # the images I use have only one color
        x[1,:,:] = x[0,:,:]
        x[2,:,:] = x[0,:,:]
        return x

def trash():
    model = AuroraClassifier()
    # instead of using the final classifier that maps feature vectors to # classes, we make this to unity such that we get out the feature vector model.fc = nn.Identity() # set model in evaluation mode (as opposed to training mode)
    model.eval()

    # path to images
    path = Path("C:/Users/lbnc/Downloads/oath_v1.1/images/")
    # load OATH dataset (with transformation defined above applied to each image as it is loaded) dataset = datasets.ImageFolder(path, Transform()) ndata = len(dataset) nfeat = 2048

    feats = np.zeros((ndata, nfeat))
    # all images in one batch, without shuffelling loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False) for step, (y, _) in enumerate(loader):
    feats[step,:] = model(y).detach().numpy().flatten()
    #break
    np.save(Path.cwd() / "features" / "feats_incept_V2.npy", feats)
