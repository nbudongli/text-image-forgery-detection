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
max_size = 512
re_size = 1024
def preprocess_mask(mask):
    mask = mask.astype(np.float32)
    mask = mask[:,:,0:1]
    mask[mask<=127.5] = 0.0
    mask[mask>127.5] = 255.
    #mask[mask<=127.5] = 255.
    #mask[mask>127.5] = 0.0
    return mask

class UNetDataset(Dataset):
    def __init__(self, dir_train, dir_mask,train_transform=None,val_transform=None,mode = 'train'):
        self.dirTrain = dir_train
        self.dirMask = dir_mask
        self.mode = mode
        self.dataTrain = [os.path.join(self.dirTrain, filename)
                          for filename in os.listdir(self.dirTrain)
                          if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.tif')]
        self.dataTrain.sort()
        self.dataMask = [os.path.join(self.dirMask, filename)
                         for filename in os.listdir(self.dirMask)
                         if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.tif')]
        self.dataMask.sort()

        self.trainDataSize = len(self.dataTrain)
        self.maskDataSize = len(self.dataMask)

        self.transform1  = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        self.toTensor  = A.Compose([ToTensorV2()])
        if train_transform is not None:
            self.train_transform = train_transform
        else:
            self.train_transform = A.Compose(
                                [
                                    A.Resize(max_size,max_size,p=1),
                                    #A.RandomCrop(width=max_size, height=max_size,p=1),
                                    A.VerticalFlip(p=0.5),
                                    A.RandomRotate90(p=0.5),
                                    A.HorizontalFlip(always_apply = False,p = 0.5),
                                    #A.ImageCompression(quality_lower=50, quality_upper=100, p=0.5),
                                    #A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=False, p=0.5),
                                    #A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225), p=1),
                                    ToTensorV2(),
                                ]
                            )
        if val_transform is not None:
            self.val_transform = val_transform
        else:
            self.val_transform = A.Compose(
                                        [
                                            A.Resize(max_size,max_size,p=1),
                                            #A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225), p=1),
                                            ToTensorV2(),
                                        ]
                                    )
    
        self.kernel = np.ones((4, 4), np.uint8) 
        self.feature_chan = [128,64,32,16]
        
    def __getitem__(self, index):
        assert self.trainDataSize == self.maskDataSize
        image_filename = self.dataTrain[index]
        image = cv2.imread(image_filename)
     
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h,w,c = image.shape

        mask_filename = self.dataMask[index]
        mask = cv2.imread(mask_filename)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        mask = preprocess_mask(mask)
        #final_mask = np.concatenate([mask,edge_mask],axis=-1)
        # print(final_mask.shape)
        # exit()
        #ret, mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY_INV)
        if self.mode=='train':
            transformed = self.train_transform(image=image, mask=mask)
            image = transformed["image"]
            final_mask = transformed["mask"]

        else:
            transformed = self.val_transform(image=image, mask=mask)
            image = transformed["image"]    
            final_mask = transformed["mask"]

        image = image.float().div(255)
        final_mask = final_mask.float().div(255)
        final_mask = final_mask.permute(2, 0, 1)


        if self.mode=='train':
            return image, final_mask,image_filename
        if self.mode=='val':
            return image, final_mask,image_filename
        if self.mode=='predict':
            return image, final_mask,w,h,image_filename
        

    def __len__(self):
        return self.trainDataSize