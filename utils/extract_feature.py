import torch
import numpy as np
from torch.autograd import Variable


def fliplr(img):
    '''flip horizontal'''
    # 水平翻转
    #inv_idx == W 长的 arange ,且从大到小[w-1,w-2, ..., 0]
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    inv_idx = inv_idx.cuda()
    # index_select 功能：在 dim上 进行索引
    #  这里的作用将 w(宽) 维度 进行翻转，
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    use_PCB = False
    features = torch.FloatTensor()
    count = 0
    #加载数据
    for batch_idx,(imgs, pids, cids) in enumerate(dataloaders):
        # 加载 数据
        n, c, h, w = imgs.size()
        # 累加所有的 batch_size，即所有数据大小
        count += n
        # 本类别的所有图片，比如query == 2368 张
        print('num of datas is {}'.format(count))

        #use pcb
        #[batch_size,2048,6]
        ff = None
        if use_PCB:
            ff = torch.FloatTensor(n,2048,6).zero_().cuda() # we have six parts

        for i in range(2):
            #= = = 水平flip = = =#
            if i == 1:
                # 水平翻转图像
                imgs = fliplr(imgs)
            input_img = Variable(imgs.cuda())

            # 输出特征
            outputs = model(input_img)
            if i == 0:
                # 这个地方 应当根据 model 最终输出的feature 维度进行修改
                # 原本为 ff = torch.FloatTensor(n, 512).zero_().cuda()
                ff = torch.FloatTensor(n, outputs.size(1)).zero_().cuda()
            # ？ 为啥是512维  ff.shape --> [batch_size, 512]
            # ff == img_flip_feature + img_raw_features
            ff += outputs

        #= = = = =  norm feature = = = = = =#
        if use_PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
            # fnorm.expand_as(ff) ,fnorm ==[batch_size, 1,6] ,扩展后[batch_size, 2048,6] 且2048维数据同原1维一样
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
        # features == ff.data.cpu()
        features = torch.cat((features,ff.data.cpu()), 0)

    return features


def extract_per_feature(imgs, model):

    ff = torch.FloatTensor()
    use_PCB = False
    for i in range(2):
        # = = = 水平flip = = =#
        if i == 1:
            # 水平翻转图像
            imgs = fliplr(imgs)
        input_img = Variable(imgs.cuda())

        # 输出特征
        #[batch_size,features]
        outputs = model(input_img)

        if i == 0:
            # 这个地方 应当根据 model 最终输出的feature 维度进行修改
            # 原本为 ff = torch.FloatTensor(n, 512).zero_().cuda()
            ff = torch.FloatTensor(outputs.size(0), outputs.size(1)).zero_().cuda()

            if use_PCB:
                ff = torch.FloatTensor(outputs.size(0), outputs.size(1), outputs.size(2)).zero_().cuda()
                # ？ 为啥是512维  ff.shape --> [batch_size, 512]
        # ff == img_flip_feature + img_raw_features
        ff += outputs
    # = = = = =  norm feature = = = = = =#
    if use_PCB:
        # feature size (n,2048,6)
        # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
        # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
        # fnorm.expand_as(ff) ,fnorm ==[batch_size, 1,6] ,扩展后[batch_size, 2048,6] 且2048维数据同原1维一样
        ff = ff.div(fnorm.expand_as(ff))
        ff = ff.view(ff.size(0), -1)
    else:
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
    # ff == [batch_size,featured]
    # features == ff.data.cpu()
    features = ff.data.cpu()
    return features