from torch.utils.data import Dataset
from torchvision.utils import save_image
import os
import numpy as np
from torchvision import transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch

#############  dataset ##################
max_size_w = 512
max_size_h = 512


def preprocess_mask(mask):
    mask = mask.astype(np.float32)
    mask = mask[:,:,0:1]
    mask[mask<=127.5] = 0.0
    mask[mask>127.5] = 255.
    return mask

class UNetDataset(Dataset):
    def __init__(self, dir_train, train_transform=None,val_transform=None,mode = 'train'):
        self.dirTrain = dir_train
        self.mode = mode
        self.dataTrain = [os.path.join(self.dirTrain, filename)
                          for filename in os.listdir(self.dirTrain)
                          if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.tif')]
        #self.dataTrain.sort()


        self.trainDataSize = len(self.dataTrain)

        self.transform1  = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        self.toTensor  = A.Compose([ToTensorV2()])
        if train_transform is not None:
            self.train_transform = train_transform
        else:
            self.train_transform = A.Compose(
                                [
                                    A.Resize(max_size_h,max_size_w,p=1),
                                    A.VerticalFlip(p=0.2),
                                    A.HorizontalFlip(p = 0.2),
                                    ToTensorV2(),
                                ]
                            )
        if val_transform is not None:
            self.val_transform = val_transform
        else:
            self.val_transform = A.Compose(
                                        [
                                            A.Resize(max_size_h,max_size_w,p=1),
                                            ToTensorV2(),
                                        ]
                                    )
    
        self.kernel = np.ones((4, 4), np.uint8) 
        self.feature_chan = [128,64,32,16]
        
    def __getitem__(self, index):
        image_filename = self.dataTrain[index]
        #print(image_filename)
        if "j.jpg" in image_filename:
            classes = 0
        else:
            classes = 1
        image = cv2.imread(image_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h,w,c = image.shape


        if self.mode=='train':
            transformed = self.train_transform(image=image)
            image = transformed["image"]

        else:
            transformed = self.val_transform(image=image)
            image = transformed["image"]    

        image = image.float().div(255)


        if self.mode=='train':
            return image, classes,image_filename
        if self.mode=='val':
            return image, classes,image_filename
        if self.mode=='predict':
            return image, classes,w,h,image_filename
        

    def __len__(self):
        return self.trainDataSize