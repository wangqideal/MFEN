# coding: utf-8
"""
通过实现Grad-CAM学习module中的forward_hook和backward_hook函数
"""
import cv2
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from torch.nn import init
from models import  pose_config
from models import get_pose_net
from models import ProxyNet, PCB, ProxyNet_sub, AdjustEncoder, ResNet50_self, ResNet50_anchor
import glob


img_width = 128
img_height = 256

def img_transform(img_in, transform):
    """
    将img进行预处理，并转换成模型输入所需的形式—— B*C*H*W
    :param img_roi: np.array
    :return:
    """
    img = img_in.copy()
    img = Image.fromarray(np.uint8(img))
    img = transform(img)
    img = img.unsqueeze(0)    # C*H*W --> B*C*H*W
    return img


def img_preprocess(img_in):
    """
    读取图片，转为模型可读的形式
    :param img_in: ndarray, [H, W, C]
    :return: PIL.image
    """
    img = img_in.copy()
    img = cv2.resize(img,(img_width, img_height))
    img = img[:, :, ::-1]   # BGR --> RGB
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [0.24580306, 0.24236229, 0.2603115])
    ])
    img_input = img_transform(img, transform)
    return img_input


def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())


def farward_hook(module, input, output):
    fmap_block.append(output)


def show_cam_on_image(img, mask, out_dir,img_name):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)

    path_cam_img = os.path.join(out_dir,'GradCam_' + img_name + ".jpg")
    # path_raw_img = os.path.join(out_dir, "raw.jpg")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    cv2.imwrite(path_cam_img, np.uint8(255 * cam))
    # cv2.imwrite(path_raw_img, np.uint8(255 * img))


def comp_class_vec(ouput_vec, index=None):
    """
    计算类向量
    :param ouput_vec: tensor
    :param index: int，指定类别
    :return: tensor
    """
    if not index:
        index = np.argmax(ouput_vec.cpu().data.numpy())
    else:
        index = np.array(index)
    index = index[np.newaxis, np.newaxis]
    index = torch.from_numpy(index)
    # one_hot = torch.zeros(1, 15110).scatter_(1, index, 1)
    # print('one_hot ---> ',one_hot.shape)
    # one_hot.requires_grad = True

    class_vec = torch.sum( output)  # one_hot = 11.8605

    return class_vec


def gen_cam(feature_map, grads):
    """
    依据梯度和特征图，生成cam
    :param feature_map: np.array， in [C, H, W]
    :param grads: np.array， in [C, H, W]
    :return: np.array, [H, W]
    """
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)

    weights = np.mean(grads, axis=(1, 2))  #

    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (img_width, img_height))
    cam -= np.min(cam)
    cam = cam / np.max(cam)

    return cam


def test_keypoint():
    # weight_path = './weights/epoch_950_netWeights.pth'
    weight_path = r'C:\Users\OCEAN\Desktop\resnet_market_70.pth'
    pre_net = models.resnet50()
    pre_net.load_state_dict(torch.load(weight_path))

    # G1 = ProxyNet_sub(branch_num= 2, return_embedding=False)
    # anchor_net = ProxyNet_sub(df_config.model, backbone = pre_net, use_GeM=df_config.use_gem, branch_num= df_config.mmd_branch_num, return_embedding=True)
    # merger_layer = nn.Conv2d(2048 * 2, 2048, 1)
    pe_net = AdjustEncoder(6, 6)

    # ================ keypoint ====================#
    # keypoints_predictor = get_pose_net(pose_config, False)
    # keypoints_predictor.load_state_dict(torch.load(pose_config.TEST.MODEL_FILE,map_location=torch.device('cpu')))
    # net = keypoints_predictor

    img_ext = '.jpg'
    imgs_list = glob.glob(os.path.join(path_img, '*' + img_ext))
    count = 0
    for img_path in imgs_list:
        fmap_block = list()
        grad_block = list()
        img = cv2.imread(img_path, 1)  # 读取图像
        basename = os.path.splitext(os.path.basename(img_path))[0]
        # print(basename)
        # h, w, c = img.shape
        # print(h,w)
        # img = np.float32(cv2.resize(img, (img_width, img_height))) / 255 #为了丢到vgg16要求的224*224 先进行缩放并且归一化
        img_input = img_preprocess(img)


        # ================keypoint====================#
        net.final_layer.register_forward_hook(farward_hook)
        net.final_layer.register_backward_hook(backward_hook)

        # forward
        output = net(img_input)
        # print('output ---- > ',output.shape)
        # print("output", output)
        # print(output.size)
        # idx = np.argmax(output.cpu().data.numpy())

        # backward
        net.zero_grad()
        class_loss = comp_class_vec(output)
        class_loss.backward()

        # 生成cam
        grads_val = grad_block[0].cpu().data.numpy().squeeze()
        fmap = fmap_block[0].cpu().data.numpy().squeeze()
        # print(grads_val.shape)
        cam = gen_cam(fmap, grads_val)

        # 保存cam图片
        img_show = np.float32(cv2.resize(img, (128, 256))) / 255
        show_cam_on_image(img_show, cam, output_dir, basename)
        count += 1
        print('prcessed image No.{}'.format(count))
    print('=' * 30)
    print('total number of processed images is {} '.format(count))
    print('=' * 30)



if __name__ == '__main__':


    path_img = 'imgs'
    # path_img = r'D:\BaiduNetdiskDownload\Market-1501-v15.09.15\bounding_box_train'
    # path_net = 'E:/pytorch/self_weight/proxy/no_mmd/supply_20_0.04371843774258937.pth'
    path_net = r'C:\Users\OCEAN\Desktop\supply_70_1.0778751742131636.pth'
    # path_net = 'C:/Users/blueDam/Desktop/supply_10_0.2639466915832887.pth'
    # path_net = 'weights/weight_keypoints/pose_hrnet_w48_256x192.pth'
    # path_net = 'E:/pytorch/self_weight/60_pcb.pth'
    # path_net = 'E:/pytorch/self_weight/proxy/re-id/map-71.3,r1-89.2,pe_scf/50_10.048434934993765.pth'
    # output_dir = r'D:\BaiduNetdiskDownload\Market-1501-v15.09.15\bounding_box_train\out_imgs'
    output_dir = 'out_imgs'
    fmap_block = list()
    grad_block = list()

    # 图片读取；网络加载
    # img = cv2.imread(path_img, 1)  # H*W*C
    # img_input = img_preprocess(img)
    #====================主干网络可视化===================#
    # weight_path = './weights/epoch_950_netWeights.pth'
    # pre_net = models.resnet50()
    # net = ResNet50_self(751)
    net = ResNet50_anchor(751)
    # pre_net.load_state_dict(torch.load(weight_path,map_location=torch.device('cpu')))
    # net = ProxyNet(backbone=pre_net)
    # net = ProxyNet()
    # net = PCB(751)
    net.load_state_dict(torch.load(path_net, map_location=torch.device('cpu')))
    net.eval()

    # ================keypoint====================#
    # keypoints_predictor = get_pose_net(pose_config, False)
    # keypoints_predictor.load_state_dict(torch.load(pose_config.TEST.MODEL_FILE,map_location=torch.device('cpu')))
    # net = keypoints_predictor
    # net.eval()


    # net = models.resnet50(pretrained=True)
    # net.layer4[0].downsample[0].stride = (1, 1)
    # net.layer4[0].conv2.stride = (1, 1)
    # net.avgpool =nn.Sequential()
    # net.fc = nn.Sequential()
    # G1_state_dict



    img_ext ='.jpg'
    imgs_list = glob.glob(os.path.join(path_img,'*'+img_ext))
    count = 0
    for img_path in imgs_list:
        fmap_block = list()
        grad_block = list()
        img = cv2.imread(img_path,1) #读取图像
        basename = os.path.splitext(os.path.basename(img_path))[0]
        # print(basename)
        h,w,c = img.shape
        # print(h,w)
        # img = np.float32(cv2.resize(img, (img_width, img_height))) / 255 #为了丢到vgg16要求的224*224 先进行缩放并且归一化
        img_input = img_preprocess(img)
        # 注册hook
        # net.conv2.register_forward_hook(farward_hook)
        # net.conv2.register_backward_hook(backward_hook)

        # ====================主干网络可视化===================#
        net.model.layer4.register_forward_hook(farward_hook)
        net.model.layer4.register_backward_hook(backward_hook)

        # ================keypoint====================#
        # net.final_layer.register_forward_hook(farward_hook)
        # net.final_layer.register_backward_hook(backward_hook)

        # forward
        output, _ = net(img_input)
        # print('output ---- > ',output.shape)
        # print("output", output)
        # print(output.size)
        idx = np.argmax(output.cpu().data.numpy())

        # backward
        net.zero_grad()
        class_loss = comp_class_vec(output)
        class_loss.backward()

        # 生成cam
        grads_val = grad_block[0].cpu().data.numpy().squeeze()
        fmap = fmap_block[0].cpu().data.numpy().squeeze()
        # print(grads_val.shape)
        cam = gen_cam(fmap, grads_val)

        # 保存cam图片
        img_show = np.float32(cv2.resize(img, (128, 256))) / 255
        show_cam_on_image(img_show, cam, output_dir,basename)
        count += 1
        print('prcessed image No.{}'.format(count))
    print('='*30)
    print('total number of processed images is {} '.format(count))
    print('=' * 30)