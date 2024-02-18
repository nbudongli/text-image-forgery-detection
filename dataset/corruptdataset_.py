from torch.utils.data import Dataset
from torchvision.utils import save_image
import os
import numpy as np
from torchvision import transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
import random


from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose,ElasticTransform
)

def strong_aug(p=0.5,p2=0.7):
    return Compose([
        A.Resize(max_size_h,max_size_w,p=1),
        A.VerticalFlip(p=0.2),
        A.HorizontalFlip(p = 0.2),
        #Flip(),
        #Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=p2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=p2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            #IAAPiecewiseAffine(p=0.3),
        ], p=p2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=p2),
        
        OneOf([
            A.RandomFog(fog_coef_lower=0.3,fog_coef_upper=1,alpha_coef=0.08,always_apply=False,p=0.4),
            A.GaussianBlur(blur_limit=11,p=0.3),
            A.RandomGamma(gamma_limit=(20,20),eps=None,p=0.2)
        ], p=p2),
        HueSaturationValue(p=0.3),
        ToTensorV2(),
    ], p=p)


def strong_affine(p=0.5,p2=0.7):
    return Compose([
        OneOf([
            IAAPiecewiseAffine(p=0.1),
            #ElasticTransform(p=0.3, alpha=120,sigma=120 * 0.05, alpha_affine=120 * 0.03),
        ], p=p2),
        ToTensorV2(),
    ], p=p)

#############  dataset ##################
max_size_w = 512
max_size_h = 768

#max_size_w = 128
#max_size_h = 192
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
                                    A.Resize(max_size_h,max_size_w,p=1),
                                    #A.RandomCrop(width=max_size, height=max_size,p=1),
                                    A.VerticalFlip(p=0.2),
                                    #A.RandomRotate90(p=0.5),
                                    A.HorizontalFlip(p = 0.2),
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
                                            A.Resize(max_size_h,max_size_w,p=1),
                                            #A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225), p=1),
                                            ToTensorV2(),
                                        ]
                                    )
    
        self.kernel = np.ones((4, 4), np.uint8) 
        self.feature_chan = [128,64,32,16]
        
        self.augmentation = strong_aug(p=1)
        self.affine_augmentation = strong_affine(p=1)
        
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
        mask_with_img = np.concatenate([mask,image],axis=-1)
        #print(mask_with_img.shape)
        #exit()
        if self.mode=='train':
            # transformed = self.train_transform(image=image, mask=mask)

            transformed = self.augmentation(image=image, mask=mask_with_img)
            distort_image = transformed["image"]
       
            mask_with_img = transformed["mask"]
            real_img = mask_with_img[:,:,1:]
            final_mask = mask_with_img[:,:,:1]
            
            # affine_transformed = self.affine_augmentation(image=distort_image, mask=final_mask)
            # distort_image = affine_transformed["image"]
            # final_mask = affine_transformed["mask"]
            
            # toTensor = self.toTensor(image=real_img)
            # real_img = toTensor["image"]



        else:
            transformed = self.val_transform(image=image, mask=mask)
            image = transformed["image"]    
            final_mask = transformed["mask"]


        if self.mode=='train':
            distort_image = distort_image.float().div(255)
            real_img = real_img.float().div(255)
            final_mask = final_mask.float().div(255)
            real_img = real_img.permute(2, 0, 1)
            final_mask = final_mask.permute(2, 0, 1)

            return distort_image, real_img,final_mask,image_filename
        
        if self.mode=='val':
            return image, final_mask,image_filename
        if self.mode=='predict':
            return image, final_mask,w,h,image_filename
        

    def __len__(self):
        return self.trainDataSize