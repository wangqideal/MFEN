import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import cfg as pose_config
from .pose_hrnet import get_pose_net
from .pose_processor import HeatmapProcessor2, HeatmapProcessor


class ScoremapComputer(nn.Module):

    def __init__(self, gaussion_smooth=None,norm_scale=None):
        super(ScoremapComputer, self).__init__()

        # init skeleton model
        self.keypoints_predictor = get_pose_net(pose_config, False)


        self.keypoints_predictor.load_state_dict(torch.load(pose_config.TEST.MODEL_FILE,map_location=torch.device('cpu')))
        # self.heatmap_processor = HeatmapProcessor(normalize_heatmap=True, group_mode='sum', gaussion_smooth=gaussion_smooth,norm_scale=10.0)
        self.heatmap_processor = HeatmapProcessor2(normalize_heatmap=True, group_mode='sum', norm_scale=norm_scale)

    def forward(self, x):
        heatmap = self.keypoints_predictor(x)  # before normalization
        # input image is [bs,3,256,128] keypoints_predictor out is： [bs, 17, 64, 32]
        scoremap, keypoints_confidence, keypoints_location = self.heatmap_processor(heatmap)  # after normalization
        return scoremap.detach(), keypoints_confidence.detach(), keypoints_location.detach()


def compute_local_features(weight_global_feature, feature_maps, score_maps, keypoints_confidence,pre_features):
    '''
    the last one is global feature
    :param config:
    :param feature_maps:
    :param score_maps:
    :param keypoints_confidence:
    :param pre_features [bs,2048]
    :return:
    '''
    # feature_maps: [bs,2048,16,8]
    fbs, fc, fh, fw = feature_maps.shape
    sbs, sc, sh, sw = score_maps.shape
    assert fbs == sbs and fh == sh and fw == sw

    # get feature_vector_list
    feature_vector_list = []
    for i in range(sc + 1):
        if i < sc:  # skeleton-based local feature vectors
            score_map_i = score_maps[:, i, :, :].unsqueeze(1).repeat([1, fc, 1, 1])
            feature_vector_i = torch.sum(score_map_i * feature_maps, [2, 3])
            print('$'*30)
            print(feature_vector_i.shape)
            feature_vector_list.append(feature_vector_i)
        else:  # global feature vectors
            #feature_maps[bs,2048,16,8]
            feature_vector_i = (
                        F.adaptive_avg_pool2d(feature_maps, 1) + F.adaptive_max_pool2d(feature_maps, 1)).squeeze()
            if feature_maps.size(0) == 1:
                feature_vector_i = torch.stack([pre_features,feature_vector_i.unsqueeze(0)],dim=2)
            else :
                assert pre_features.shape == feature_vector_i.shape , "feature's shape is no match"
                feature_vector_i = torch.stack([pre_features,feature_vector_i],dim=2)

            # feature_vector_i [bs,2048]
            print('$'*30)
            print(feature_vector_i.shape)
            feature_vector_list.append(feature_vector_i)
            keypoints_confidence = torch.cat([keypoints_confidence, torch.ones([fbs, 1]).cuda()], dim=1)

    # compute keypoints confidence
    keypoints_confidence[:, sc:] = F.normalize(
        keypoints_confidence[:, sc:], 1, 1) * weight_global_feature  # global feature score_confidence
    keypoints_confidence[:, :sc] = F.normalize(keypoints_confidence[:, :sc], 1,
                                               1) * weight_global_feature  # partial feature score_confidence

    return feature_vector_list, keypoints_confidence
