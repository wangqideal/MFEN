import os
import cv2
import numpy as np
from utils import load_file_clazz

# roor_dir = 'E:/pytorch/dataset/edu_dataset/origin-img'
roor_dir = 'E:/pytorch/dataset/edu_dataset/origin-img'
img_dir ='com_imgs'
# img_dir = 'Img'
label_dir = 'Gt'
img_ext = '.jpg'
label_ext = '.png'


# ================ mean ====================#
# channel of r mean is--- 0.4199877907132573，g mean is-- 0.40242363746270676，r mean is-- 0.39834195933747135
# channel of r std is--- 0.20417425922510724，g std is-- 0.20237039203647686，r std is-- 0.1976008507639633
def cal_mean_std():
    b_img_mean_list=[]
    g_img_mean_list=[]
    r_img_mean_list=[]
    b_img_std_list=[]
    g_img_std_list=[]
    r_img_std_list=[]


    img_path_list=load_file_clazz(roor_dir,img_dir,img_ext)

    for idx,img_path in enumerate(img_path_list):
        mean=[]
        std=[]
        img=cv2.imread(img_path)

        mean,std = cv2.meanStdDev(img / 255)

        # b_img_mean,g_img_mean,r_img_mean,_=cv2.mean(img/255)
        b_img_mean_list.append(mean[0])
        g_img_mean_list.append(mean[1])
        r_img_mean_list.append(mean[2])

        b_img_std_list.append(std[0])
        g_img_std_list.append(std[1])
        r_img_std_list.append(std[2])

    r_mean=np.sum(r_img_mean_list)/len(r_img_mean_list)
    g_mean=np.sum(g_img_mean_list)/len(g_img_mean_list)
    b_mean=np.sum(b_img_mean_list)/len(b_img_mean_list)

    r_std = np.sum(r_img_std_list) / len(r_img_std_list)
    g_std = np.sum(g_img_std_list) / len(g_img_std_list)
    b_std = np.sum(b_img_std_list) / len(b_img_std_list)

    print('len of r_mean_list is-- {}，g_mean_list is-- {}，b_mean_list is-- {}'.format(len(r_img_mean_list),len(g_img_mean_list),len(b_img_mean_list)))
    print('channel of r mean is--- {}，g mean is-- {}，r mean is-- {}'.format(r_mean,g_mean,b_mean))
    print('channel of r std is--- {}，g std is-- {}，r std is-- {}'.format(r_std, g_std, b_std))
    return r_mean,g_mean,b_mean,r_std, g_std, b_std


if __name__ == '__main__':
    r_m,g_m,b_m,r_std,g_std,b_std=cal_mean_std()

