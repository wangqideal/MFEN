from matplotlib import pyplot as plt
import cv2
import numpy as np
from PIL import Image
from models import pose_config, get_pose_net,ScoremapComputer,AdjustEncoder
import torch
from torchvision import transforms as T
import visdom


# viz = visdom.Visdom(env='heatmap_test')



def BlendHeatmap(img, heatmaps, joint_num):
    '''data:original photo
    heatmaps: heatmap of all 19 joints(channels),Array shape(128,228,19)
    joint_num:heatmap of which joint(channel)to visualize'''

    ch, h_heatmap, w_heatmap = heatmaps.shape

    heatmap = heatmaps[joint_num, :, : ]*255.0
    # resize
    scaled_img = cv2.resize(heatmap, (128, 256), interpolation=cv2.INTER_CUBIC)
    # print(scaled_img.shape)
    # blend resized image and heatmap
    plt.imshow(img, alpha=1)
    plt.imshow(scaled_img, alpha=0.65)
    # add colorbar of the heatmap
    plt.colorbar(fraction=0.04, pad=0.03)

def preprocessing(img,h,w):
    image =  np.copy(img)
    image = cv2.resize(image,(w,h))
    image = image.transpose((2,0,1))
    image = image.reshape(1,3,h,w)
    return image
def tensor_to_np(tensor):

    img = tensor.mul(255).byte()
    img = img.cpu().numpy()
    return img

def test():

    weight_path = r'C:\Users\blueDam\Desktop\PROXY\keypoints_noAM_30.pth'
    img_path = './imgs/0802_c1_f0204448.jpg'
    img = Image.open(img_path)
    self_transform = T.Compose([
        T.Resize((256, 128)),
        T.ToTensor()
    ])
    img_1 = self_transform(img)
    img_1 = torch.unsqueeze(img_1, dim=0)
    scoremap_computer = ScoremapComputer(norm_scale=10)
    scoremap_computer.eval()

    score_maps, keypoints_confidence, max_index = scoremap_computer(img_1)
    print(keypoints_confidence)
    adjust = AdjustEncoder(6,6)



    model_weight = torch.load(weight_path,map_location=torch.device('cpu'))
    adjust.load_state_dict(model_weight['G2_state_dict'])
    adjust.eval()


    adj_score_maps = adjust(score_maps)
    score_maps_temp = adj_score_maps.detach() * 255.0

    score_maps_temp = (score_maps_temp).numpy()

    raw_img = cv2.imread(img_path)
    h,w,c = raw_img.shape

    out_heatmap = np.zeros([score_maps.size(1),h,w])

    for j in range(score_maps.size(1)):
        out_heatmap[j] = cv2.resize(score_maps_temp[0][j],(w,h))

    for c in range(len(out_heatmap)):
        out_heatmap[c] =  np.where(out_heatmap[c]>0,out_heatmap[c],0)

    out_heatmap = np.sum(out_heatmap,axis=0)
    img_bgr = raw_img
    b,g,r = cv2.split(img_bgr)
    img_rgb = cv2.merge([r,g,b])

    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(img_rgb, alpha=1)
    plt.imshow(out_heatmap, alpha=0.85)

if __name__ == '__main__':
    test()
    plt.show()

