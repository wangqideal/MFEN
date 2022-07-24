from torch import nn
from torchvision import models
import torch


class Encoder(nn.Module):

    def __init__(self, class_num):
        super(Encoder, self).__init__()

        self.class_num = class_num

        # backbone and optimize its architecture
        resnet = models.resnet50(pretrained=True)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)

        # cnn backbone
        self.resnet_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.maxpool,  # no relu
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        # self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        feature_map = self.resnet_conv(x)
        return feature_map



if __name__ == '__main__':
    model = Encoder(751)
    temp = torch.rand(3,3,256,128)
    result = model(temp)
    print(result.shape)

