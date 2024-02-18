import cv2
import numpy as np
import os

if __name__ == '__main__':
  
  root_dir = '/share/home/dongli/Liang/DL_code/ali_com/new_dataset/final_dataset_masks_1/'
  save_edge_dir =  '/share/home/dongli/Liang/DL_code/ali_com/new_dataset/final_dataset_mask_edge_1/'
  if os.path.exists(save_edge_dir)==False:
      os.makedirs(save_edge_dir) 
      
  file_list = os.listdir(root_dir)
  for i in range(len(file_list)):
    file_path = os.path.join(root_dir,file_list[i])
    img = cv2.imread(file_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge_output = cv2.Canny(gray, 32, 128)
    kernel = np.ones((4, 4), np.uint8)
    img_dilate = cv2.dilate(edge_output, kernel) # , iteration = 1
    save_path =  os.path.join(save_edge_dir,file_list[i])
    cv2.imwrite(save_path,img_dilate)
    print(save_path)
  