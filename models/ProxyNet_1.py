from torchvision import models,transforms
import torch
from torch import nn
from torch.nn import functional as F


class Fallten(nn.Module):
    def __init__(self):
        super(Fallten, self).__init__()
    def forward(self,x):
        return x.view(x.size(0),-1)

# def weights_init_kaiming(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
#     elif classname.find('Linear') != -1:
#         nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
#         nn.init.constant_(m.bias.data, 0.0)
#     elif classname.find('BatchNorm1d') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0.0)

# def weights_init_classifier(m):
#     classname = m.__class__.__name__
#     if classname.find('Linear') != -1:
#         nn.init.normal_(m.weight.data, std=0.001)
#         nn.init.constant_(m.bias.data, 0.0)
def weights_init_kaiming(m):
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

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class Embedder_BottleBlock(nn.Module):
    def __init__(self, in_dim, out_dim, relu=True, dropout=True, bottle_dim=1024):
        super(Embedder_BottleBlock, self).__init__()

        bottle = [nn.Linear(in_dim, bottle_dim)]
        bottle += [nn.BatchNorm1d(bottle_dim)]
        if relu:
            bottle += [nn.LeakyReLU(0.1)]
        if dropout:
            bottle += [nn.Dropout(p=0.5)]
        bottle = nn.Sequential(*bottle)
        bottle.apply(weights_init_kaiming)
        self.bottle = bottle

        classifier = [nn.Linear(bottle_dim, out_dim)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self, x):
        x = self.bottle(x)
        x = self.classifier(x)
        return x
class BottleClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, relu=True, dropout=True, bottle_dim=512):
        super(BottleClassifier, self).__init__()

        bottle = [nn.Linear(in_dim, bottle_dim)]
        bottle += [nn.BatchNorm1d(bottle_dim)]
        if relu:
            bottle += [nn.LeakyReLU(0.1)]
        if dropout:
            bottle += [nn.Dropout(p=0.5)]
        bottle = nn.Sequential(*bottle)
        bottle.apply(weights_init_kaiming)
        self.bottle = bottle

        classifier = [nn.Linear(bottle_dim, out_dim)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self, x):
        x = self.bottle(x)
        x = self.classifier(x)
        return x

class BottleBlock(nn.Module):
    def __init__(self, in_dim, out_dim, relu=True, dropout=True, bottle_dim=1024):
        super(BottleBlock, self).__init__()

        bottle = [nn.Linear(in_dim, bottle_dim)]
        bottle += [nn.BatchNorm1d(bottle_dim)]
        if relu:
            bottle += [nn.LeakyReLU(0.1)]
        if dropout:
            bottle += [nn.Dropout(p=0.5)]
        bottle = nn.Sequential(*bottle)
        bottle.apply(weights_init_kaiming)
        self.bottle = bottle

        classifier = [nn.Linear(bottle_dim, out_dim)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self, x):
        x = self.bottle(x)
        x = self.classifier(x)
        return x

class Predictor(nn.Module):
    def __init__(self, in_dim, out_dim, relu=True, dropout=True, bottle_dim=1024):
        super(Predictor, self).__init__()

        bottle = [nn.Linear(in_dim, bottle_dim)]
        bottle += [nn.BatchNorm1d(bottle_dim)]
        if relu:
            bottle += [nn.LeakyReLU(0.1)]
        if dropout:
            bottle += [nn.Dropout(p=0.5)]
        bottle = nn.Sequential(*bottle)
        bottle.apply(weights_init_kaiming)
        self.bottle = bottle

        classifier = [nn.Linear(bottle_dim, out_dim)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self, x):
        x = self.bottle(x)
        x = self.classifier(x)
        return x
# BN -- Liner
class BNClassifier(nn.Module):

    def __init__(self, in_dim, class_num):
        super(BNClassifier, self).__init__()

        self.in_dim = in_dim
        self.class_num = class_num

        self.bn = nn.BatchNorm1d(self.in_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.in_dim, self.class_num, bias=False)

        self.bn.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        feature = self.bn(x)
        cls_score = self.classifier(feature)
        return  cls_score


class BNClassifiers(nn.Module):

    def __init__(self, in_dim, class_num, branch_num):
        super(BNClassifiers, self).__init__()

        self.in_dim = in_dim
        self.class_num = class_num
        self.branch_num = branch_num

        for i in range(self.branch_num):
            setattr(self, 'classifier_{}'.format(i), BNClassifier(self.in_dim, self.class_num))

    def __call__(self, feature_vector_list):

        assert len(feature_vector_list) == self.branch_num

        # bnneck for each sub_branch_feature
        cls_score_list = []
        for i in range(self.branch_num):
            feature_vector_i = feature_vector_list[i]
            classifier_i = getattr(self, 'classifier_{}'.format(i))
            # bned_feature_vector_i, cls_score_i = classifier_i(feature_vector_i)
            cls_score_i = classifier_i(feature_vector_i)
            # bned_feature_vector_list.append(bned_feature_vector_i)
            cls_score_list.append(cls_score_i)

        return cls_score_list

class ProxyNet_2(nn.Module):
    def __init__(self, backbone1, backbone2, in_ch=3, out_ch=1):
        super(ProxyNet_2, self).__init__()

        self.model_name = self.__class__.__name__
        self.global_channel = 2
        self.class_num = 751

        # remove the final downsample
        backbone2.layer4[0].downsample[0].stride = (1, 1)
        backbone2.layer4[0].conv2.stride = (1, 1)

        backbone1.layer4[0].downsample[0].stride = (1, 1)
        backbone1.layer4[0].conv2.stride = (1, 1)

        base_net_1 = nn.Sequential(*list(backbone1.children())[:-2])
        base_net_2 = nn.Sequential(*list(backbone2.children())[:-2])
        self.backbone_1 = base_net_1
        self.backbone_2 = base_net_2

        self.set_grad(self.backbone_1, False)

        self.dropout = nn.Dropout(p=0.5)
        self.avgpool_MAP_local = nn.AdaptiveAvgPool2d((2, 1))
        # self.avgpool_PCB = nn.AdaptiveAvgPool2d((self.part, 1))
        self.avgpool_MAP_global = nn.AdaptiveAvgPool2d((self.global_channel, 1))
        self.flatten = Fallten()

        # x bottle block
        # for i in range(self.map_channel):
        #     name = 'embedder_x' + str(i)
        #     setattr(self, name, BottleBlock(2048, 2048, relu=True, dropout=False, bottle_dim=512))

        setattr(self, 'embedder_x', Embedder_BottleBlock(2048 * 2, 2048, relu=True, dropout=False, bottle_dim=1024))

        # xx bottle block
        setattr(self, 'embedder_xx', Embedder_BottleBlock(2048 * 2, 2048, relu=True, dropout=False, bottle_dim=1024))

        # ===========PCB==============#

        # modules = list(resnet.children())[:-2]
        # self.backbone = nn.Sequential(*modules)
        # self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        #
        #
        # # PCB define 6 classifiers
        # self.classifiers = nn.ModuleList()
        # for i in range(self.part):
        #     self.classifiers.append(ClassBlockPCB(2048, self.class_num, droprate=0.5, relu= False, bnorm=True, num_bottleneck=256))

    def set_grad(self, model, val=False):
        for p in model.parameters():
            p.requires_grad = val

    def forward(self, x):

        # torch.Size([3, 2048, 8, 4])
        x_pre = self.backbone_1(x)
        x_feature = self.backbone_2(x)
        # print(x_feature.shape)
        # print(x_pre.shape)
        # torch.Size([3, 2048, 8, 4])

        # ============未进入 embeding ===============#
        # [batch,2048,2,1]
        x_pre_embeding_pool = self.avgpool_MAP_local(x_pre)
        x_embeding_pool = self.avgpool_MAP_local(x_feature)
        features = x_embeding_pool

        # ============= end ==============#

        # =============pooling==============#
        # [batch,2048,2,1]
        # features = self.avgpool_MAP_global(x_feature)
        # [batch,2048,2]
        features = torch.squeeze(features)
        features_global = torch.FloatTensor().cuda()
        for i in range(self.global_channel):
            features_global = torch.cat([features_global, features[:, :, i]], dim=1)
        # [batch,2048*2]

        # ============映射===============#  # 线性还是非线性？？？？
        # [batch_size, 2048*2]
        x_pre_embeding_pool = torch.squeeze(x_pre_embeding_pool)
        t1, t2 = torch.chunk(x_pre_embeding_pool, 2, dim=2)
        x_pre_embeding_pool = torch.cat([torch.squeeze(t1), torch.squeeze(t2)], dim=1)
        embedder_x = getattr(self, 'embedder_x')
        embedding_x_pre = embedder_x(x_pre_embeding_pool)
        # [batch_size, 2048]

        # xx
        # [batch_size,2048，2]
        x_embeding_pool = torch.squeeze(x_embeding_pool)
        t1, t2 = torch.chunk(x_embeding_pool, 2, dim=2)
        x_embeding_pool = torch.cat([torch.squeeze(t1), torch.squeeze(t2)], dim=1)
        embedder_xx = getattr(self, 'embedder_xx')
        embedding_x = embedder_xx(x_embeding_pool)
        # [batch_size,2048]
        # ===========end================#

        # ===========================#
        # [batch_size,2048*2] + [batch_size,2048*2]
        # features_embedding = torch.cat([x_pre_embeding_pool, x_embeding_pool], dim=1)
        # features_embedding_2 = torch.stack([x_pre_embeding_pool, x_embeding_pool], dim=2)
        # [batch_size,2048*4]

        # ================mmd============#
        embedding = torch.stack([embedding_x_pre, embedding_x], dim=2)
        # [batch_size,2048,2]
        # =========end==========#

        features_all = torch.cat([features_global, x_pre_embeding_pool], dim=1)
        # [batch_size, 2048 * 4]

        if self.training:
            # print(result_feature.shape)
            # features_global: [batch_size,2048*2]  features_embedding；[batch_size,2048*2]  embedding： [batch_size, 2048, 2]
            return features_all, embedding
        else:
            return features_all
            # return F.normalize(all_feature,p=2,dim=2).view(-1,2048*20)


class ProxyNet_1(nn.Module):
    def __init__(self,backbone1, backbone2, in_ch=3, out_ch=1, one_classifier =False):
        super(ProxyNet_1, self).__init__()
        self.one_classifier = one_classifier
        self.map_channel = 8
        self.global_channel = 4
        self.class_num = 751

        backbone2.layer4[0].downsample[0].stride = (1, 1)
        backbone2.layer4[0].conv2.stride = (1, 1)

        backbone1.layer4[0].downsample[0].stride = (1, 1)
        backbone1.layer4[0].conv2.stride = (1, 1)

        base_net_1 = nn.Sequential(*list(backbone1.children())[:-2])
        base_net_2 = nn.Sequential(*list(backbone2.children())[:-2])

        self.backbone_1 = base_net_1
        self.backbone_2 = base_net_2
        self.set_grad(self.backbone_1,False)
        self.flatten = Fallten()

        # x bottle block
        for i in range(self.map_channel):
            name = 'embedder_x' + str(i)
            setattr(self, name, BottleBlock(2048*8, 2048, relu=True, dropout=False, bottle_dim=512))

        # xx bottle block
        for i in range(self.map_channel):
            name = 'embedder_xx' + str(i)
            setattr(self, name, BottleBlock(2048*8, 2048, relu=True, dropout=False, bottle_dim=512))

        # global bottle block
        for i in range(self.global_channel):
            name = 'embedder_global' + str(i)
            setattr(self, name, BottleBlock(2048*8, 2048, relu=True, dropout=False, bottle_dim=1024))

        # classifier
        for i in range(20):
            name = 'classifier' + str(i)
            setattr(self, name, BottleClassifier(2048, self.class_num, relu=True, dropout=False, bottle_dim=512))

        self.classifier = BottleClassifier(2048*20, self.class_num,  relu=True,dropout=False,bottle_dim=1024)

    def set_grad(self, model, val=False):
        for p in model.parameters():
            p.requires_grad = val

    def forward(self,x):

        # torch.Size([3, 2048, 8, 4])
        x_pre = self.backbone_1(x)
        x_global = self.backbone_2(x)
        print(x_pre.shape)
        print(x_global.shape)
        xx = x_global - x_pre

        # torch.Size([3, 2048, 4, 8])
        global_map = x_global.permute(0,1,3,2)

        # torch.Size([3, 2048, 8, 8])
        x_pre_map = x_pre @ global_map
        xx_map = xx @ global_map



        #  x
        embedding_cat_x = None
        for i in range(self.map_channel):
            if self.map_channel == 1:
                features_i_x = x_pre_map
            else:
                features_i_x = self.flatten(x_pre_map[:,:,:,i])
            embedder_i_x = getattr(self, 'embedder_x'+str(i))
            embedding_i_x = embedder_i_x(features_i_x)
            if i == 0:
                embedding_cat_x = embedding_i_x
            else:
                embedding_cat_x = torch.stack([embedding_cat_x,embedding_i_x],dim=2)
        # [batch_size,2048,8]


        # xx
        embedding_cat_xx = None
        for i in range(self.map_channel):
            if self.map_channel == 1:
                features_i_xx = xx_map
            else:
                features_i_xx = self.flatten(xx_map[:,:,:,i])
            embedder_i_xx = getattr(self, 'embedder_xx'+str(i))
            embedding_i_xx = embedder_i_xx(features_i_xx)
            if i == 0:
                embedding_cat_xx = embedding_i_xx
            else:
                embedding_cat_xx = torch.stack([embedding_cat_xx,embedding_i_xx],dim=2)
        #[batch_size,2048,8]


        # global
        embedding_cat_global = None
        for i in range(self.global_channel):
            if self.global_channel == 1:
                features_i_global = x_global
            else:
                features_i_global = self.flatten(x_global[:,:,:,i])
            embedder_i_global = getattr(self, 'embedder_global'+str(i))
            embedding_i_global = embedder_i_global(features_i_global)
            if i == 0:
                embedding_cat_global = embedding_i_global
            else:
                embedding_cat_global = torch.stack([embedding_cat_global,embedding_i_global],dim=2)
        #[batch_size,2048,4]


        all_feature_middle = torch.cat([embedding_cat_x,embedding_cat_xx],dim=2)
        all_feature = torch.cat([all_feature_middle,embedding_cat_global],dim=2)
        #[batch_size,2048,20]

        logits_list = []
        result_feature = None
        if self.one_classifier:
            result_feature = self.classifier(all_feature)
        else:
            for i in range(20):
                features_i = all_feature_middle[:, :, i]
                classifier_i = getattr(self, 'classifier' + str(i))
                logits_i = classifier_i(features_i)
                logits_list.append(logits_i)




        # [batch_size, 2048*(8+8+4)]

        # print(result_feature.shape)
        if self.one_classifier:
            return result_feature, embedding_cat_x, embedding_cat_xx
        else:
            return logits_list, embedding_cat_x, embedding_cat_xx


class ProxyNet(nn.Module):
    def __init__(self, backbone1, backbone2, in_ch=3, out_ch=1):
        super(ProxyNet, self).__init__()

        self.model_name = self.__class__.__name__
        self.global_channel = 2
        self.class_num = 751

        # remove the final downsample
        backbone2.layer4[0].downsample[0].stride = (1, 1)
        backbone2.layer4[0].conv2.stride = (1, 1)

        backbone1.layer4[0].downsample[0].stride = (1, 1)
        backbone1.layer4[0].conv2.stride = (1, 1)

        base_net_1 = nn.Sequential(*list(backbone1.children())[:-2])
        base_net_2 = nn.Sequential(*list(backbone2.children())[:-2])
        self.backbone_1 = base_net_1
        self.backbone_2 = base_net_2

        self.set_grad(self.backbone_1, False)

        self.dropout = nn.Dropout(p=0.5)
        self.avgpool_MAP_local = nn.AdaptiveAvgPool2d((1, 1))
        # self.avgpool_PCB = nn.AdaptiveAvgPool2d((self.part, 1))
        self.avgpool_MAP_global = nn.AdaptiveAvgPool2d((self.global_channel, 1))
        self.flatten = Fallten()

        # x bottle block
        # for i in range(self.map_channel):
        #     name = 'embedder_x' + str(i)
        #     setattr(self, name, BottleBlock(2048, 2048, relu=True, dropout=False, bottle_dim=512))

        setattr(self, 'embedder_x', Embedder_BottleBlock(2048, 2048, relu=True, dropout=False, bottle_dim=512))

        # xx bottle block
        setattr(self, 'embedder_xx', Embedder_BottleBlock(2048, 2048, relu=True, dropout=False, bottle_dim=512))

        # ===========PCB==============#

        # modules = list(resnet.children())[:-2]
        # self.backbone = nn.Sequential(*modules)
        # self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        #
        #
        # # PCB define 6 classifiers
        # self.classifiers = nn.ModuleList()
        # for i in range(self.part):
        #     self.classifiers.append(ClassBlockPCB(2048, self.class_num, droprate=0.5, relu= False, bnorm=True, num_bottleneck=256))

    def set_grad(self, model, val=False):
        for p in model.parameters():
            p.requires_grad = val

    def forward(self, x):

        # torch.Size([3, 2048, 8, 4])
        x_pre = self.backbone_1(x)
        x_feature = self.backbone_2(x)
        # print(x_feature.shape)
        # print(x_pre.shape)
        # torch.Size([3, 2048, 8, 4])

        # ============未进入 embeding ===============#
        # [batch,2048,2,1]
        x_pre_embeding_pool = self.avgpool_MAP_local(x_pre)
        x_embeding_pool = self.avgpool_MAP_local(x_feature)
        # features = x_embeding_pool

        # ============= end ==============#

        # =============pooling==============#
        # [batch,2048,2,1]
        features = self.avgpool_MAP_global(x_feature)
        features = self.dropout(features)
        # [batch,2048,2]
        features = torch.squeeze(features)
        features_global = torch.FloatTensor().cuda()
        for i in range(self.global_channel):
            features_global = torch.cat([features_global, features[:, :, i]], dim=1)
        # [batch,2048*2]

        # ============映射===============#  # 线性还是非线性？？？？
        # [batch_size, 2048*2]
        x_pre_embeding_pool = torch.squeeze(x_pre_embeding_pool)
        # t1,t2 = torch.chunk(x_pre_embeding_pool,2,dim=2)
        # x_pre_embeding_pool = torch.cat([torch.squeeze(t1),torch.squeeze(t2)],dim=1)
        embedder_x = getattr(self, 'embedder_x')
        embedding_x_pre = embedder_x(x_pre_embeding_pool)
        # [batch_size, 2048]

        # xx
        # [batch_size,2048，2]
        x_embeding_pool = torch.squeeze(x_embeding_pool)
        # t1,t2 = torch.chunk(x_embeding_pool,2,dim=2)
        # x_embeding_pool = torch.cat([torch.squeeze(t1),torch.squeeze(t2)],dim=1)
        embedder_xx = getattr(self, 'embedder_xx')
        embedding_x = embedder_xx(x_embeding_pool)
        # [batch_size,2048]
        # ===========end================#

        # ===========================#
        # [batch_size,2048*2] + [batch_size,2048*2]
        # features_embedding = torch.cat([x_pre_embeding_pool, x_embeding_pool], dim=1)
        # features_embedding_2 = torch.stack([x_pre_embeding_pool, x_embeding_pool], dim=2)
        # [batch_size,2048*4]

        # ================mmd============#
        embedding = torch.stack([embedding_x_pre, embedding_x], dim=2)
        # [batch_size,2048,2]
        # =========end==========#

        features_all = torch.cat([features_global, embedding_x_pre], dim=1)
        # [batch_size, 2048 * 3]

        if self.training:
            # print(result_feature.shape)
            # features_global: [batch_size,2048*2]  features_embedding；[batch_size,2048*2]  embedding： [batch_size, 2048, 2]
            return features_all, embedding
        else:
            return features_all
            # return F.normalize(all_feature,p=2,dim=2).view(-1,2048*20)


class DisTill_unit(nn.Module):
    def __init__(self, backbone1, backbone2):
        super(DisTill_unit, self).__init__()

        self.model_name = self.__class__.__name__
        self.global_channel = 2
        self.class_num = 751

        # remove the final downsample
        backbone2.layer4[0].downsample[0].stride = (1, 1)
        backbone2.layer4[0].conv2.stride = (1, 1)

        backbone1.layer4[0].downsample[0].stride = (1, 1)
        backbone1.layer4[0].conv2.stride = (1, 1)

        base_net_1 = nn.Sequential(*list(backbone1.children())[:-2])
        base_net_2 = nn.Sequential(*list(backbone2.children())[:-2])
        self.backbone_1 = base_net_1
        self.backbone_2 = base_net_2

        self.set_grad(self.backbone_1, False)

        self.dropout = nn.Dropout(p=0.5)
        self.avgpool_embedder = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_MAP_global = nn.AdaptiveAvgPool2d((self.global_channel, 1))

        setattr(self, 'embedder_x', Embedder_BottleBlock(2048, 2048, relu=True, dropout=False, bottle_dim=512))

        # xx bottle block
        setattr(self, 'embedder_xx', Embedder_BottleBlock(2048, 2048, relu=True, dropout=False, bottle_dim=512))

    def set_grad(self, model, val=False):
        for p in model.parameters():
            p.requires_grad = val

    def forward(self, x):

        # torch.Size([3, 2048, 16, 8])
        x_pre = self.backbone_1(x)
        x_feature = self.backbone_2(x)


        # ============未进入 embeding ===============#
        # x_pre_embeding_pool [batch,2048,]
        # x_embeding_pool [batch,2048,1,1]
        # x_pre_embeding_pool = self.avgpool_embedder(x_pre)
        x_pre_features = (F.adaptive_avg_pool2d(x_pre, 1) + F.adaptive_max_pool2d(x_pre, 1)).squeeze()
        x_embeding_pool = self.avgpool_embedder(x_feature).squeeze()
        # features = x_embeding_pool
        # ============= end ==============#



        # ============映射===============#  # 非线性？？？？
        # t1,t2 = torch.chunk(x_pre_embeding_pool,2,dim=2)
        # x_pre_embeding_pool = torch.cat([torch.squeeze(t1),torch.squeeze(t2)],dim=1)
        embedder_x = getattr(self, 'embedder_x')
        embedding_x_pre = embedder_x(x_pre_features)
        # [batch_size, 2048]

        # xx
        # x_embeding_pool [batch_size,2048]
        # t1,t2 = torch.chunk(x_embeding_pool,2,dim=2)
        # x_embeding_pool = torch.cat([torch.squeeze(t1),torch.squeeze(t2)],dim=1)
        embedder_xx = getattr(self, 'embedder_xx')
        embedding_x = embedder_xx(x_embeding_pool)
        # [batch_size,2048]
        # ===========end================#


        # ================mmd============#
        embedding = torch.stack([embedding_x_pre, embedding_x], dim=2)
        # embedding : [batch_size,2048,2]
        # =========end==========#

        # 这里要不要加一个id 损失 与mmd形成对抗，保证pre 和 主干网络的feature 对分类有作用
        return x_feature, embedding


class AdjustEncoder(nn.Module):
    def __init__(self, in_ch, out_ch,dirate=1):
        super(AdjustEncoder, self).__init__()

        self.model_name = self.__class__.__name__

        self.rebnconv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )





    def forward(self,x):

        conv_1 = self.rebnconv(x)
        conv_2 = self.rebnconv(conv_1)
        conv_3 = self.rebnconv(conv_2)

        return conv_3

if __name__ == '__main__':
    temp = torch.rand(4,3,16,8)
    net = AdjustEncoder(3,3)
    net(temp)
    # weight_path = '../weights/epoch_950_netWeights.pth'
    # restNet50 = models.resnet50(pretrained=True)
    # pre_model = models.resnet50()
    # pre_model.load_state_dict(torch.load(weight_path,map_location='cpu'))
    # net = ProxyNet_1(pre_model,restNet50)
    # temp = torch.rand(3,3,256,128)
    # # temp = temp.cuda()
    # # with open('proxy.txt','w') as f:
    # #     f.write(str(net))
    # #     f.flush()
    # # net.cuda()
    # result = net(temp)

    # bottle = BottleClassifier(2048,751)
    # with open('bottle.txt', 'w') as f:
    #     f.write(str(bottle))
    #     f.flush()