import torch as t
from torchvision import models
from torch import nn
import glob
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from models import AdjustEncoder
from models import get_pose_net





img_height = 256
img_width = 128

class FeatureExtractor():

    def __init__(self, model, target_layers):
        self.model = model
        # self.model_features = model.features
        # self.model_features = model
        self.target_layers = target_layers
        self.gradients = list()

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def get_gradients(self):
        return self.gradients

    def __call__(self, x):
        # print(self.target_layers)
        target_activations = list()
        self.gradients = list()

        for name, module in self.model._modules.items():
            print('mudole --- > ', module)
            print('x -----> ', x.shape)
            x = module(x)

            if name in self.target_layers:
                print('name --->  ', name)
                x.register_hook(self.save_gradient)
                target_activations += [x]

        return target_activations, x
def preprocess_image(img):

    mean = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    preprocessed_img = img.copy()[:, :, ::-1]  # BGR > RGB

    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - mean[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]

    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))  # transpose HWC > CHW
    preprocessed_img = t.from_numpy(preprocessed_img)  # totensor
    preprocessed_img.unsqueeze_(0)
    input = t.tensor(preprocessed_img, requires_grad=True)

    return input

def show_cam_on_image(img, mask, img_name):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite('GradCam_{}.jpg'.format(img_name), np.uint8(255 * cam))

    cam = cam[:, :, ::-1]  # BGR > RGB
    # plt.figure(figsize=(10, 10))
    # plt.imshow(np.uint8(255 * cam))

class GradCam():

    def __init__(self, model, target_layer_names):
        self.model = model
        self.model.eval()

        self.extractor = FeatureExtractor(self.model, target_layer_names)

    def forward(self, input):
        print('===== forward=====')
        return self.model(input)

    def __call__(self, input):
        # print('=====call ======')
        # print(input.shape)
        features, output = self.extractor(input)
        # output.data

        one_hot = output.max()

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grad_val = self.extractor.get_gradients()[-1].data.numpy()


        target = features[-1]
        target = target.data.numpy()[0, :]  # (1, 512, 14, 14) > (512, 14, 14)

        weights = np.mean(grad_val, axis=(2, 3))[0, :]

        cam = np.zeros(target.shape[1:])

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]


        cam = cv2.resize(cam, (img_width, img_height))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


def vis_model_struct(model):

    with open('model_name.txt','w') as f:
        f.write(str(model))
        f.flush()
def test():
    img = cv2.imread('./human.jpg')
    h, w, c = img.shape
    print(h, w)
    img_result = cv2.resize(img,(320,320))
    cv2.imshow('dd',img_result)
    cv2.waitKey(0)

from models import  pose_config
if __name__ == '__main__':

    keypoints_predictor = get_pose_net(pose_config, False)
    keypoints_predictor.load_state_dict(t.load(pose_config.TEST.MODEL_FILE,map_location=t.device('cpu')))
    # net = nn.Sequential(*list(net.children())[:-2])
    # with open('model_construct.txt','w') as f:
    #     f.write(str(keypoints_predictor))
    #     f.flush()





    # model_path = './49_(0.0017156975769264452, 0.0036634024618075692).pth'
    # model_path = './16_(0.1689806432723616, 0.10996901914629009).pth'
    # model_path ='./weights/u2net_human_seg.pth'
    # model_path = 'E:/pytorch/self_weight/epoch_950_netWeights.pth'
    # model_path = 'E:/pytorch/self_weight/proxy/mmd/core_40_10mmd.pth'
    # model_path = 'C:/Users/blueDam/Desktop/supply_0.pth'

    # model = U2NET_OFFICAL()
    # # model = VGG_SELF(3,10)
    # model.load_state_dict(t.load(model_path, map_location=t.device('cpu')))
    # net = models.resnet50()
    # net.layer4[0].downsample[0].stride = (1, 1)
    # net.layer4[0].conv2.stride = (1, 1)
    # net.avgpool =nn.Sequential()
    # net.fc = nn.Sequential()
    # #
    #
    net.load_state_dict(t.load(model_path, map_location=t.device('cpu')),strict=False)
    # keypoint : final_layer
    #layer4
    grad_cam = GradCam(model = net, target_layer_names = ["layer4"])

    img_ext ='.jpg'
    imgs_list = glob.glob(os.path.join('imgs','*'+img_ext))
    for img_path in imgs_list:
        img = cv2.imread(img_path)
        basename = os.path.splitext(os.path.basename(img_path))[0]
        print(basename)
        h,w,c = img.shape
        print(h,w)
        img = np.float32(cv2.resize(img, (img_width, img_height))) / 255
        input = preprocess_image(img)
        # print(input.shape)
        mask = grad_cam(input)
        print(mask.shape)
        print(img.shape)
        show_cam_on_image(img, mask, basename)