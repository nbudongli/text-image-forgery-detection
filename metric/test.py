
import os
import numpy as np
from Metric import *
import cv2

def cal_score(val_img_dir,save_dir):
    dir_1 = save_dir
    dir_2 = val_img_dir

    file_list_1 = os.listdir(dir_1)
    file_list_2 = os.listdir(dir_2)
    file_list_1.sort()
    file_list_2.sort()
    f1_list = []
    iou_list = []
    for i in range(len(file_list_1)):
        file_path_1 = os.path.join(dir_1,file_list_1[i])
        file_path_2 = os.path.join(dir_2,file_list_2[i])

        img_1 = cv2.imread(file_path_1,cv2.IMREAD_GRAYSCALE)
        img_2 = cv2.imread(file_path_2,cv2.IMREAD_GRAYSCALE)
        h,w = img_1.shape
        h_,w_ = img_2.shape
        if h!=h_ or w!=w_:
            img_2 = cv2.resize(img_2,(w,h))
        img_1[img_1>127] = 255
        img_1[img_1<127] = 0
        img_2[img_2>127] = 255
        img_2[img_2<127] = 0
        f1,iou = metric_numpy(img_1,img_2)
        f1_list.append(f1)
        iou_list.append(iou)
    f1_avg = np.mean(f1_list)     
    iou_avg = np.mean(iou_list)
    score = f1_avg + iou_avg
    return score,f1_avg,iou_avg

