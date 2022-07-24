import torch
import numpy as np
from utils import  NumpyCatMeter,  CMCWithVer
from models import compute_local_features



def testwithVer2(config, G1, G2, scoremap_computer, anchor_net, merger_layer, queryloader, galleryloader, bned_feature_vector_list, use_gpu, viz):
    G1.eval()
    G2.eval()
    scoremap_computer.eval()
    anchor_net.eval()
    merger_layer.eval()

    # meters
    query_features_meter, query_features2_meter, query_pids_meter, query_cids_meter = NumpyCatMeter(), NumpyCatMeter(), NumpyCatMeter(), NumpyCatMeter()
    gallery_features_meter, gallery_features2_meter, gallery_pids_meter, gallery_cids_meter = NumpyCatMeter(), NumpyCatMeter(), NumpyCatMeter(), NumpyCatMeter()

    # init dataset
    loaders = [queryloader, galleryloader]
    # compute query and gallery features
    with torch.no_grad():
        for loader_id, loader in enumerate(loaders):
            for data in loader:
                # compute feautres
                images, pids, cids = data
                images, pids, cids = images.cuda(), pids.cuda(), cids.cuda()

                # ========== test =========== #
                # feature_maps, torch.Size([bs, 2048, 16, 8])
                feature_maps, _ = G1(images)
                feature_maps_supply, _ = anchor_net(images)
                # feature_maps：torch.Size([1, 2048, 16, 8])
                # score_maps: [BS, 13, 16, 8]， keypoints_confidence:[BS, 13]，max_index：[BS, 17, 2]
                score_maps, keypoints_confidence, _ = scoremap_computer(images)
                # ===========adjust==============#
                score_maps = G2(score_maps)
                # =========== end ==============#

                # =========== 合并特征 ==============#
                feature_maps = merger_layer(torch.cat([feature_maps, feature_maps_supply], dim=1))
                # =========== end ==============#

                # feature_vector_list.len == 14  local.len()==13 [bs,2048] ,global.len() ==1 [bs,2048,2];
                # keypoints_confidence == torch.Size([1, 14])
                feature_vector_list, keypoints_confidence = compute_local_features(
                    config.weight_global_feature, feature_maps, score_maps, keypoints_confidence)


                bs, keypoints_num = keypoints_confidence.shape  # keypoints_confidence [bs,14]
                keypoints_confidence = torch.sqrt(keypoints_confidence).unsqueeze(2).repeat([1, 1, 2048]).view(
                    [bs, 2048 * keypoints_num])

                features_stage1 = keypoints_confidence * torch.cat(bned_feature_vector_list, dim=1)
                features_stage2 = torch.cat([i.unsqueeze(1) for i in bned_feature_vector_list], dim=1)


                # save as query features
                if loader_id == 0:
                    query_features_meter.update(features_stage1.data.cpu().numpy())
                    query_features2_meter.update(features_stage2.data.cpu().numpy())
                    query_pids_meter.update(pids.cpu().numpy())
                    query_cids_meter.update(cids.cpu().numpy())
                # save as gallery features
                elif loader_id == 1:
                    gallery_features_meter.update(features_stage1.data.cpu().numpy())
                    gallery_features2_meter.update(features_stage2.data.cpu().numpy())
                    gallery_pids_meter.update(pids.cpu().numpy())
                    gallery_cids_meter.update(cids.cpu().numpy())

    #
    query_features = query_features_meter.get_val()
    query_features2 = query_features2_meter.get_val()
    gallery_features = gallery_features_meter.get_val()
    gallery_features2 = gallery_features2_meter.get_val()

    # compute mAP and rank@k
    query_info = (query_features, query_features2, query_cids_meter.get_val(), query_pids_meter.get_val())
    gallery_info = (gallery_features, gallery_features2, gallery_cids_meter.get_val(), gallery_pids_meter.get_val())

    alpha =  1.0
    topk = 8
    mAP, cmc = CMCWithVer()(query_info, gallery_info,topk=topk,alpha=alpha)

    return mAP, cmc
