import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torchvision import models
from torchsummary import summary
from thop import profile
model = models.resnet50(pretrained=True)

######################################################################


class Fallten(nn.Module):
    def __init__(self):
        super(Fallten, self).__init__()
    def forward(self,x):
        return x.view(x.size(0),-1)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)



def bn_weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def bn_weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x,f]
        else:
            x = self.classifier(x)
            return x

class BNClassifier(nn.Module):

    def __init__(self, in_dim, class_num):
        super(BNClassifier, self).__init__()

        self.in_dim = in_dim
        self.class_num = class_num

        self.bn = nn.BatchNorm1d(self.in_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.in_dim, self.class_num, bias=False)

        self.bn.apply(bn_weights_init_kaiming)
        self.classifier.apply(bn_weights_init_classifier)

    def forward(self, x):
        feature = self.bn(x)
        cls_score = self.classifier(feature)
        return  cls_score

# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, circle=False):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.circle = circle
        self.classifier = ClassBlock(2048, class_num, droprate, return_f = circle)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


# Part Model proposed in Yifan Sun etal. (2018)
class PCB(nn.Module):
    def __init__(self, class_num):
        super(PCB, self).__init__()
        self.model_name = self.__class__.__name__
        self.class_num = class_num
        self.use_bn_cls = False
        self.part = 6 # We cut the pool5 to 6 parts
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        # define 6 classifiers
        if self.use_bn_cls:
            for i in range(self.part):
                name = 'classifier' + str(i)
                setattr(self, name,
                        BNClassifier(2048, self.class_num))
        else:
            for i in range(self.part):
                name = 'classifier'+str(i)
                setattr(self, name, ClassBlock(2048, self.class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        # torch.Size([8, 2048, 6, 1])
        # print(x.shape)
        predict = list()
        # get six part feature batchsize*2048*6
        features = torch.squeeze(x)
        for i in range(self.part):
            # part[i] = x[:,:,i].view(x.size(0), x.size(1))
            part_i = features[:,:,i]
            name = 'classifier'+str(i)
            c = getattr(self,name)
            predict.append(c(part_i))

        if self.training:
            return predict
        else:
            return features


class ResNet50_self(nn.Module):
    def __init__(self, class_num):
        super(ResNet50_self, self).__init__()
        self.model_name = self.__class__.__name__
        self.class_num = class_num
        self.use_bn_cls = False
        self.use_GeM = False
        self.part = 1  # We cut the pool5 to 6 parts
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool_MAP = nn.AdaptiveAvgPool2d((self.part, 1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)
        # define 6 classifiers


        if self.use_bn_cls:
            name = 'classifier'
            setattr(self, name,
                    BNClassifier(2048, self.class_num))
        else:
            name = 'classifier'
            setattr(self, name,
                    ClassBlock(2048, self.class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        features = x
        # x = self.avgpool(x)

        # torch.Size([8, 2048, 6, 1])
        # print(x.shape)
        # predict = list()
        # get six part feature batchsize*2048*6
        # features = torch.squeeze(x)

        if self.use_GeM:
            x = (F.adaptive_avg_pool2d(x, (self.part, 1)) + F.adaptive_max_pool2d(x,
                                                                                        (self.part, 1))).squeeze()
        else:
            x = self.avgpool_MAP(x).squeeze()


        x = self.dropout(x)

        if self.training:
            return getattr(self, 'classifier')(x)
        else:
            return features
class ResNet50_anchor(nn.Module):
    def __init__(self, class_num):
        super(ResNet50_anchor, self).__init__()
        self.model_name = self.__class__.__name__
        self.class_num = class_num
        self.use_bn_cls = False
        self.use_GeM = False
        self.part = 1  # We cut the pool5 to 6 parts
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool_MAP = nn.AdaptiveAvgPool2d((self.part, 1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)
        # define 6 classifiers


        # if self.use_bn_cls:
        #     name = 'classifier'
        #     setattr(self, name,
        #             BNClassifier(2048, self.class_num))
        # else:
        #     name = 'classifier'
        #     setattr(self, name,
        #             ClassBlock(2048, self.class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        # x = self.avgpool(x)

        # torch.Size([8, 2048, 6, 1])
        # print(x.shape)
        # predict = list()
        # get six part feature batchsize*2048*6
        # features = torch.squeeze(x)
        features = x
        if self.use_GeM:
            x = (F.adaptive_avg_pool2d(x, (self.part, 1)) + F.adaptive_max_pool2d(x,(self.part, 1))).squeeze()
        else:
            x = self.avgpool_MAP(x).squeeze()

        if self.training:
            x = self.dropout(x)


        # predict = getattr(self, 'classifier')(x)

        return features, x

        # if self.training:
        #     return predict
        # else:
        #     return features,x
class PCB_am(nn.Module):
    def __init__(self, class_num):
        super(PCB_am, self).__init__()
        self.model_name = self.__class__.__name__
        self.class_num = class_num
        self.use_bn_cls = False
        self.part = 6  # We cut the pool5 to 6 parts
        self.branch_num = 2
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft

        weight_path = './weights/epoch_950_netWeights.pth'
        pre_net = models.resnet50()
        pre_net.load_state_dict(torch.load(weight_path))

        self.pre_model = pre_net

        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.avgpool_MAP = nn.AdaptiveAvgPool2d((self.branch_num,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)

        self.pre_model.layer4[0].downsample[0].stride = (1, 1)
        self.pre_model.layer4[0].conv2.stride = (1, 1)

        self.merger_layer = nn.Conv2d(2048*2,2048,1)
        self.fallten =  Fallten()
        self.set_grad(self.pre_model,False)

        setattr(self, 'embedder_x', BNClassifier(2048 * self.branch_num, 2048 * self.branch_num))
        setattr(self, 'embedder_x_pre', BNClassifier(2048 * self.branch_num, 2048 * self.branch_num))


        # define 6 classifiers
        if self.use_bn_cls:
            for i in range(self.part):
                name = 'classifier' + str(i)
                setattr(self, name,
                        BNClassifier(2048, self.class_num))
        else:
            for i in range(self.part):
                name = 'classifier' + str(i)
                setattr(self, name,
                        ClassBlock(2048, self.class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

    def set_grad(self, model, val=False):
        for p in model.parameters():
            p.requires_grad = val

    def forward(self, x):
        x_pre = x
        x = self.model.conv1(x)
        x_pre = self.pre_model.conv1(x_pre)

        x = self.model.bn1(x)
        x_pre = self.pre_model.bn1(x_pre)

        x = self.model.relu(x)
        x_pre = self.pre_model.relu(x_pre)

        x = self.model.maxpool(x)
        x_pre = self.pre_model.maxpool(x_pre)

        x = self.model.layer1(x)
        x_pre = self.pre_model.layer1(x_pre)

        x = self.model.layer2(x)
        x_pre = self.pre_model.layer2(x_pre)

        x = self.model.layer3(x)
        x_pre = self.pre_model.layer3(x_pre)

        x = self.model.layer4(x)
        x_pre = self.pre_model.layer4(x_pre)

        # merger [bs,4096,16,8]
        merger = torch.cat([x,x_pre],dim=1)
        # merger [bs,2048,16,8]
        merger = self.merger_layer(merger)

        x_em = self.avgpool_MAP(x).squeeze()
        x_pre_em = self.avgpool_MAP(x_pre).squeeze()


        embedder_x_pre = getattr(self, 'embedder_x_pre')
        embedding_x_pre = embedder_x_pre(self.fallten(x_pre_em))

        embedder_x = getattr(self, 'embedder_x')
        embedding_x = embedder_x(self.fallten(x_em))

        # embedding_x_pre = torch.randn_like(embedding_x_pre)
        embedding = torch.stack([embedding_x_pre, embedding_x],dim=2)

        # x = self.avgpool(x)
        x = self.avgpool(merger)
        x = self.dropout(x)
        # torch.Size([8, 2048, 6, 1])
        # print(x.shape)
        predict = list()
        # get six part feature batchsize*2048*6
        features = torch.squeeze(x)
        for i in range(self.part):
            # part[i] = x[:,:,i].view(x.size(0), x.size(1))
            part_i = features[:, :, i]
            name = 'classifier' + str(i)
            c = getattr(self, name)
            predict.append(c(part_i))

        if self.training:
            return predict , embedding
        else:
            return features

class PCB_test(nn.Module):
    def __init__(self,model):
        super(PCB_test,self).__init__()
        self.part = 6
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0),x.size(1),x.size(2))
        return y
'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it. 
    net = PCB(751)
    print(summary(net, input_size=[(3, 256, 128)], device='cpu'))
    # net.classifier = nn.Sequential()
    # print(net)
    # input =torch.rand(8, 3, 256, 128)
    # output = net(input)
    # print('net output size:')
    # print(output[0].shape)

    # temp_list = list()
    # temp = torch.rand(1, 3, 256, 128)
    # temp_list.append(temp)
    # flops, params = profile(net, inputs=temp_list)
    # print('FLOPS -->', flops / (1000 ** 3))
    # print('Params-->', params / (1000 ** 2))
