import os
import errno
import json
import glob
import torch
from torch.autograd import Variable
import numpy as np
from models import compute_local_features

#创建文件
def mkdir_if_missing(dir):
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
def load_file_clazz(root_dir,img_dir,lab_dir,img_ext,lab_ext):
    imgs_list = glob.glob(os.path.join(root_dir,img_dir,'*' + img_ext))
    labs_list = glob.glob(os.path.join(root_dir,lab_dir,'*' + lab_ext))
    return imgs_list, labs_list

#返回模型大小 mb
def get_model_size(path,_type='mb'):
    if _type=='kb':
        return str(round(os.stat(path).st_size / 1024,3)) +' KB'
    elif _type=='mb':
        return str(round(os.stat(path).st_size / 1024/1024,3)) + ' MB'

#一次读取一行 json 数据
def read_json(fpath):
    with open(fpath,'r') as f:
        obj = json.load(f)
    return obj
def write_json(obj,fpath):
    mkdir_if_missing(os.path.join(fpath))
    with open(fpath,'w') as f:
        json.dump(obj,f,indent=4,separators=(',',':'))



def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    inv_idx = inv_idx.cuda()
    img_flip = img.index_select(3,inv_idx)

    return img_flip

def extract_per_feature__(imgs, model,config):

    ff = torch.FloatTensor()
    use_PCB = config.use_pcb
    for i in range(2):
        if i == 1:
            imgs = fliplr(imgs)
        input_img = Variable(imgs.cuda())
        outputs = model(input_img)

        if i == 0:
            ff = torch.FloatTensor(outputs.size(0), outputs.size(1)).zero_().cuda()

            if use_PCB:
                ff = torch.FloatTensor(outputs.size(0), outputs.size(1), outputs.size(2)).zero_().cuda()
        # ff == img_flip_feature + img_raw_features
        ff += outputs
    # = = = = =  norm feature = = = = = =#
    if use_PCB:
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
        ff = ff.div(fnorm.expand_as(ff))
        ff = ff.view(ff.size(0), -1)
    else:
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
    # ff == [batch_size,featured]
    # features == ff.data.cpu()
    features = ff.data.cpu()
    return features

def extract_per_feature(imgs, model, config):

    ff = torch.FloatTensor()
    use_PCB = config.use_pcb
    use_proxy = config.use_proxy
    for i in range(2):
        if i == 1:
            imgs = fliplr(imgs)
        input_img = Variable(imgs.cuda())

        outputs = model(input_img)

        if i == 0:
            ff = torch.FloatTensor(outputs.size(0), outputs.size(1)).zero_().cuda()
            if use_PCB:
                ff = torch.FloatTensor(outputs.size(0), outputs.size(1), outputs.size(2)).zero_().cuda()
            if use_proxy:
                ff = torch.FloatTensor(outputs.size(0), outputs.size(1), outputs.size(2)).zero_().cuda()
        # ff == img_flip_feature + img_raw_features
        ff += outputs
    # = = = = =  norm feature = = = = = =#
    if use_PCB:
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
        ff = ff.div(fnorm.expand_as(ff))
        ff = ff.view(ff.size(0), -1)
    elif use_proxy:
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(3)
        ff = ff.div(fnorm.expand_as(ff))
        ff = ff.view(ff.size(0), -1)
    else:
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
    features = ff.data.cpu()
    return features

def extract_per_feature_proxy(imgs, G1, scoremap_computer, config):

    ff = torch.FloatTensor()
    use_PCB = config.use_PCB
    proxy = config.use_proxy
    for i in range(2):
        # = = = 水平flip = = =#
        if i == 1:
            # 水平翻转图像
            imgs = fliplr(imgs)
        input_img = Variable(imgs.cuda())

        # 输出特征
        # ========== test =========== #
        # feature_maps, torch.Size([bs, 2048, 16, 8])
        # embedding  [bs, 2048, 2]
        feature_maps, embedding = G1(input_img)
        pre_features = embedding[:, :, 0]
        # feature_maps：torch.Size([1, 2048, 16, 8])
        # score_maps: [BS, 13, 16, 8]， keypoints_confidence:[BS, 13]，max_index：[BS, 17, 2]
        score_maps, keypoints_confidence, _ = scoremap_computer(imgs)
        # feature_vector_list.len == 14  local.len()==13 [bs,2048] ,global.len() ==1 [bs,2048,2];
        # keypoints_confidence == torch.Size([1, 14])
        feature_vector_list, keypoints_confidence = compute_local_features(
            config.weight_global_feature, feature_maps, score_maps, keypoints_confidence, pre_features)
        # feature_vector_list_local.len() == 13 ,item == [bs, 2048]
        feature_vector_list_local = feature_vector_list[:13]
        # feature_vector_list_supply [bs, 2048*2]
        feature_vector_list_supply = (feature_vector_list[-1]).view(feature_vector_list[-1].size(0), -1)

        bs, keypoints_num = keypoints_confidence.shape  # keypoints_confidence [bs,14]
        #one_tensor[bs,1]
        one_tensor = torch.ones(bs,1)
        # keypoints_confidence_test[bs,15]
        keypoints_confidence_test = torch.cat([keypoints_confidence,one_tensor],dim=1)
        # keypoints_confidence_test[bs,15*2048]
        keypoints_confidence_test = torch.sqrt(keypoints_confidence_test).unsqueeze(2).repeat([1, 1, 2048]).view(
            [bs, 2048 * keypoints_num+1])

        #feature_vector_test [bs,15*2048]
        feature_vector_test = torch.cat([torch.cat(feature_vector_list_local, dim=1), feature_vector_list_supply],
                                        dim=1)

        # feature_vector_test[bs, 15 * 2048]
        features_stage1 = keypoints_confidence_test * feature_vector_test
        features_satge2 = torch.cat([i.unsqueeze(2) for i in feature_vector_test], dim=2)
        # features_satge2 [bs,2048,15]

        # ================= end =====================#

        if i == 0:
            # 这个地方 应当根据 model 最终输出的feature 维度进行修改
            # 原本为 ff = torch.FloatTensor(n, 512).zero_().cuda()
            ff = torch.FloatTensor(features_satge2.size(0), features_satge2.size(1),features_satge2.size(2)).zero_().cuda()

            if proxy:
                ff = torch.FloatTensor(features_satge2.size(0), features_satge2.size(1), features_satge2.size(2)).zero_().cuda()
                # ？ 为啥是512维  ff.shape --> [batch_size, 512]
        # ff == img_flip_feature + img_raw_features
        ff += features_satge2
    # = = = = =  norm feature = = = = = =#
    if use_PCB:
        # feature size (n,2048,6)
        # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
        # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
        # fnorm.expand_as(ff) ,fnorm ==[batch_size, 1,6] ,扩展后[batch_size, 2048,6] 且2048维数据同原1维一样
        ff = ff.div(fnorm.expand_as(ff))
        ff = ff.view(ff.size(0), -1)
    elif proxy:
        # feature size (n,2048,15)
        # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
        # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(15)
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