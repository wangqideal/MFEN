import torch
import os
import time
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import itertools
from torchvision.transforms import transforms as T
from torchvision import models
from torch.utils.data import DataLoader, random_split
from data import ImageDataset
import torch.nn.functional as F

from data import init_img_dataset,get_dataset_name
from torchnet.meter import AverageValueMeter, ConfusionMeter
from torch.backends import cudnn
from configs import config_Proxy_Keypoints
from utils import evaluate, Visualier, mkdir_if_missing, mmd, extract_per_feature, extract_per_feature_proxy,extract_per_feature_proxy_with_G2, evaluate_gpu,CrossEntropyLabelSmooth, TripletLoss
from models import init_model,Predictor, ScoremapComputer, compute_local_features,  BNClassifiers, BNClassifier, AdjustEncoder, Predictors
from test_val import testwithVer2

df_config = config_Proxy_Keypoints()
torch.manual_seed(df_config.seed)

device = None
use_gpu =False

def ide_dis_factor(condfid_list):
    factors = torch.exp(condfid_list ** 2) - 1
    # print(factors)
    # [bs,14]
    for i in range(factors.size(0)):
        for j in range(factors.size(1)):
            if factors[i, j] > 1.0:
                factors[i, j] = 1.0
    return factors

def compute_ide_loss(ide_creiteron, score_list, pids, weights):
    loss_all = 0
    # print(weights.shape)
    # weights = ide_dis_factor(weights)
    # print('=' * 30)
    # print(weights)
    for i, score_i in enumerate(score_list):
        loss_i = ide_creiteron(score_i, pids)
        # loss_i = (weights[:, i] * loss_i).mean( )
        loss_i = (loss_i).mean()
        loss_all += loss_i
    return loss_all

def compute_triplet_loss(triplet_creiteron, feature_list, pids):
    '''we suppose the last feature is global, and only compute its loss'''
    loss_all = 0
    for i, feature_i in enumerate(feature_list):
        if i == len(feature_list) - 1:
            loss_i = triplet_creiteron(feature_i, feature_i, feature_i, pids, pids, pids)
            loss_all += loss_i
    return loss_all

# distance of F1 F2
def discrepancy(out1, out2):
    return torch.mean(torch.abs(torch.softmax(out1,dim=1) - torch.softmax(out2,dim=1)))
# consine distance
def cosine_dist(x, y):
    '''
    :param x: torch.tensor, 2d
    :param y: torch.tensor, 2d
    :return:
    '''

    bs1 = x.size()[0]
    bs2 = y.size()[0]

    frac_up = torch.matmul(x, y.transpose(0, 1))
    frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
            (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
    cosine = frac_up / frac_down

    return cosine

def main():
    global device, use_gpu
    os.environ['CUDA_VISABLE_DEVICES'] = df_config.gpu_devices
    use_gpu = torch.cuda.is_available()
    if use_gpu and df_config.use_gpu:
        use_gpu = True
    else:
        use_gpu = False
    print('## = = = = = = = = = = = = = = = = ##')

    if use_gpu:
        print("INFO  : % - - - We will use GPU to run - - - %")
        print("The Host has {} GPUs".format(torch.cuda.device_count()))
        print("Currently using GPU {}".format(df_config.gpu_devices))
        #cudnn
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(df_config.seed)
        device = torch.device('cuda',int(df_config.gpu_devices))
    else:
        # pass
        print("Currently using CPU (GPU is highly recommended)")

    pin_memory = True if use_gpu else False


    #============== loading data ===============#

    transform_train = T.Compose([
        T.Resize((df_config.in_height,df_config.in_width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = T.Compose([
        T.Resize((df_config.in_height, df_config.in_width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Initializing dataset {}".format(df_config.dataset))
    dataset = init_img_dataset(name=df_config.dataset, root_dir=df_config.root_dir,img_ext=df_config.img_ext,
                     split_id=df_config.split_id, cuhk03_labeled=df_config.cuhk03_labeled, cuhk03_classic_split=df_config.cuhk03_classic_split,
                     )
    len_all_train_data = len(dataset.train)

    #val 与 train  2：8
    # train_data , val_data = random_split(dataset.train,[round(len_all_train_data*0.9),round(len_all_train_data*0.1)])
    print('## = = = = = = = loading data = = = = = = = = = ##')
    print('------------------num of train dataset is ：{}'.format(len_all_train_data))
    trainloader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train, is_transform=True),
        batch_size=df_config.train_batch_size, shuffle=True, num_workers=df_config.num_workers,
        pin_memory=pin_memory, drop_last=True,
    )
    # valloader = DataLoader(
    #     ImageDataset(val_data, transform=transform_train,is_transform=True),
    #     batch_size=df_config.train_batch_size, shuffle=True, num_workers=df_config.num_workers,
    #     pin_memory=pin_memory, drop_last=True,
    # )

    queryloader = DataLoader(
        ImageDataset(dataset.query, transform=transform_test,is_transform=True),
        batch_size=df_config.test_batch_size, shuffle=False, num_workers=df_config.num_workers,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, transform=transform_test,is_transform=True),
        batch_size=df_config.test_batch_size, shuffle=False, num_workers=df_config.num_workers,
        pin_memory=pin_memory, drop_last=False,
    )

    #  =================init net ==================== #
    print('## = = = = = = = loading model = = = = = = = = = ##')
    print("Initializing model")

    # weight_path = './weights/epoch_950_netWeights.pth'
    # restNet50 = models.resnet50(pretrained=True)
    # pre_model = models.resnet50()
    # pre_model.load_state_dict(torch.load(weight_path))

    # ===========  anchor net init ============== #
    weight_path = './weights/epoch_950_netWeights.pth'
    pre_net = models.resnet50()
    pre_net.load_state_dict(torch.load(weight_path))

    
    
    # G1 = init_model(df_config.model, use_GeM=df_config.use_gem, branch_num= df_config.mmd_branch_num, return_features=True)
    # merger_layer = nn.Conv2d(2048 * 2, 2048, 1)
    # anchor_net = init_model(df_config.model, backbone=pre_net, use_GeM=df_config.use_gem, branch_num= df_config.mmd_branch_num, return_features=True)
    # anchor_net.eval()

    G1 = init_model(df_config.model, use_GeM=df_config.use_gem, branch_num= df_config.mmd_branch_num, return_embedding=False)
    anchor_net = init_model(df_config.model, backbone = pre_net, use_GeM=df_config.use_gem, branch_num= df_config.mmd_branch_num, return_embedding=True)
    merger_layer = nn.Conv2d(2048 * 2, 2048, 1)
    G2 = AdjustEncoder(df_config.branch_num,df_config.branch_num)


    # init backbone
    # G1 = DisTill_unit(backbone1=pre_model, backbone2=restNet50)
    # G2 = AdjustEncoder(6,6)
    # keypoints model
    scoremap_computer = ScoremapComputer(norm_scale=df_config.norm_scale,gaussion_smooth=df_config.gaussion_smooth)
    scoremap_computer = scoremap_computer.eval()
    bnclassifiers_ide_local = BNClassifiers(2048, 702, df_config.branch_num,re_f=False)
    # bnclassifiers_ide_local = Predictors(2048, 702, df_config.branch_num)
    # bnclassifiers_ide_supply = BNClassifier(2048*2,751)
    # bnclassifiers_ide_supply = Predictor(2048*2,751)



    #use Gpu
    G1.to(device)
    merger_layer.to(device)
    anchor_net.to(device)
    G2.to(device)
    scoremap_computer.to(device)
    bnclassifiers_ide_local.to(device)
    # bnclassifiers_ide_supply.to(device)
    #  ================= end ==================== #
    # load pretrained weight
    if df_config.preTrained:
        G1.load_state_dict(torch.load(df_config.load_model_path)['G1_state_dict'])
        # anchor_net.load_state_dict(torch.load(df_config.load_model_path)['anchor_state_dict'])
        merger_layer.load_state_dict(torch.load(df_config.load_model_path)['merger_layer_state_dict'])
        # G2.load_state_dict(torch.load(df_config.load_model_path)['G2_state_dict'])
    lr = df_config.lr
    #loss
    criterion_ce = nn.CrossEntropyLoss()
    criterion_ide = CrossEntropyLabelSmooth(702,reduce=False)
    # criterion_l1 = nn.SmoothL1Loss()
    criterion_mse = nn.MSELoss()
    # criterion_triplet = TripletLoss(df_config.margin, 'euclidean')

    #optim
    # optimizer_G = torch.optim.Adam(itertools.chain(G1.parameters(),merger_layer.parameters()), lr=lr, weight_decay=df_config.weight_decay)
    optimizer_G = torch.optim.Adam(G1.parameters(), lr=lr, weight_decay=df_config.weight_decay)
    optimizer_G2 = torch.optim.Adam(G2.parameters(), lr=lr, weight_decay=df_config.weight_decay)
    optimizer_C1 = torch.optim.Adam(bnclassifiers_ide_local.parameters(), lr=lr, weight_decay=df_config.weight_decay)
    # optimizer_C2 = torch.optim.Adam(bnclassifiers_ide_supply.parameters(), lr=lr, weight_decay=df_config.weight_decay)


    #update lr
    lr_scheduler_G=None
    lr_scheduler_G2=None
    lr_scheduler_C1 = None
    # lr_scheduler_C2 = None

    if df_config.step_size>0:
        lr_scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_G, step_size=df_config.step_size, gamma=df_config.gamma)
        lr_scheduler_G2 = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_G2, step_size=df_config.step_size, gamma=df_config.gamma)
        lr_scheduler_C1 = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_C1, step_size=df_config.step_size, gamma=df_config.gamma)
        # lr_scheduler_C2 = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_C2, step_size=df_config.step_size, gamma=df_config.gamma)

    #init visdom
    viz=Visualier(env=df_config.env, port=df_config.port)
    #statistic
    loss_all = AverageValueMeter()
    loss_G2_mse = AverageValueMeter()
    G2_beta= AverageValueMeter()
    loss_pe = AverageValueMeter()
    loss_mmd_local_vector = AverageValueMeter()
    loss_mmd_pre = AverageValueMeter()
    loss_ide_local = AverageValueMeter()

    # loss_triplet_supply = AverageValueMeter()



    # ============= train ================#
    print('#======== starting  train =========#')
    best_rank1 = -np.inf
    best_epoch = 0
    spend_all_time = 0
    for epoch in tqdm(range(df_config.max_epoch)):
        # init
        start_time_per_epoch = time.time()

        loss_all.reset()
        loss_mmd_local_vector.reset()
        loss_mmd_pre.reset()
        loss_ide_local.reset()
        loss_pe.reset()
        G2_beta.reset()
        loss_G2_mse.reset()
        # loss_triplet_supply.reset()

        # switch mode
        G1.train()
        G2.train()
        bnclassifiers_ide_local.train()
        # bnclassifiers_ide_supply.train()


        for batch_idx, (imgs, pids, cids) in enumerate(trainloader):

            if use_gpu:
                imgs, pids = imgs.to(device), pids.to(device)


            # ==========  train  =========== #

            #==============input===============#
            # feature_maps, torch.Size([bs, 2048, 16, 8])
            # mmd_embedder_core  [bs, 2048, 2]
            # feature_maps_supply, torch.Size([bs, 2048, 16, 8])
            # mmd_embedder_supply  [bs, 2048*2]
            feature_maps, mmd_embedder_core = G1(imgs)
            feature_maps_supply, mmd_embedder_supply = anchor_net(imgs)
            # score_maps: [1, 13, 16, 8]， keypoints_confidence:[1, 13]，max_index：[1, 17, 2]
            score_maps, keypoints_confidence, _ = scoremap_computer(imgs)
            score_maps_temp = score_maps
            # print('='*30)
            # print(keypoints_confidence)
            # print(keypoints_confidence.sum(dim=1))
            # a = 3/0
            # =============== beta ================ #
            # beta = ((keypoints_confidence.sum(dim=1) / 6.0  ).sum() / df_config.train_batch_size)

            min_idx = torch.argmin(keypoints_confidence, dim=1)
            max_idx = torch.argmax(keypoints_confidence, dim=1)

            sum_counter_list = list()
            for i in range(keypoints_confidence.size(0)):
                keypoints_confidence_temp = keypoints_confidence[i, :]
                sum_counter = 0
                for j in range(len(keypoints_confidence_temp)):
                    if (j == min_idx[i]) or (j == max_idx[i]):
                        continue
                    sum_counter += keypoints_confidence_temp[j]
                sum_counter_list.append(sum_counter / ( keypoints_confidence.size(1) - 2.0))
            beta = sum(sum_counter_list) / keypoints_confidence.size(0)

            # print(beta)
            # =========== end ==============#


            


            #=========== merge feature ==============#
            # [bs,2048*2,16,8]
            # feature_maps = torch.cat([feature_maps,feature_maps_supply],dim=1)
            # feature_maps = merger_layer(feature_maps)
            # =========== end ==============#

            #===========adjust==============#
            score_maps_adjust = G2(score_maps)
            # score_maps = score_maps + (beta * score_maps_adjust)
            # score_maps = score_maps + score_maps_adjust
            score_maps = score_maps_adjust
            # score_maps = G2(score_maps)



            #=========== end ==============#

            # =========== align keypoints  ==============#
            # feature_vector_list.len == 14   item==[bs,2048] 
            # keypoints_confidence == torch.Size([1, 14])
            feature_vector_list, keypoints_confidence = compute_local_features(
                 df_config.weight_global_feature, feature_maps, score_maps, keypoints_confidence)
            # =========== end ==============#

            # feature_vector_list_local = feature_vector_list[:13]
            # feature_vector_list_supply [bs,2048*2]
            # feature_vector_list_supply = torch.cat([feature_vector_list[-1][:,:,0], feature_vector_list[-1][:,:,1]], dim=1)
            # feature_vector_list_supply = (feature_vector_list[-1]).view(feature_vector_list[-1].size(0),-1)
            feature_vector_list_local = feature_vector_list[:-1]
            # bned_feature_vector_list, cls_score_list_local = bnclassifiers_ide_local(feature_vector_list)
            cls_score_list_local = bnclassifiers_ide_local(feature_vector_list_local)

            # cls_score_list_supply = bnclassifiers_ide_supply(feature_vector_list_supply.view(feature_vector_list_supply.size(0),-1))
            # cls_score_list_local.append(cls_score_list_supply)
            # ================= end =====================#


            # =============== loss ================== #
            ide_loss_local = compute_ide_loss(criterion_ide, cls_score_list_local, pids, keypoints_confidence)
            # ide_loss_supply = 1.0*criterion_ide(cls_score_list_supply,pids).mean()
            # triplet_loss_supply = criterion_triplet(feature_vector_list_supply, feature_vector_list_supply, feature_vector_list_supply, pids, pids, pids)
            # ================= end =====================#

            # ================= mmd =====================#

            feature1 = torch.cat([mmd_embedder_core[:, :, i] for i in range(mmd_embedder_core.size(2))], dim=1)
            # feature2 = torch.cat([mmd_embedder_supply[:, :, i] for i in range(mmd_embedder_supply.size(2))], dim=1)

            # ======= anchor Net && backbone ==============#
            pre_mmd_loss = mmd(feature1, mmd_embedder_supply)
            # ================= end =====================#

            #===============pe loss ================#
            mse_loss = criterion_mse(score_maps_adjust, score_maps_temp)
            # mse_loss = 0
            # pe_loss = beta * mse_loss + (1 - beta) * ide_loss_local
            pe_loss = beta * mse_loss
            # pe_loss =  mse_loss
            #=======================================#

            # ================per_channel mmd====================#
            per_local_vector_mmd_loss = 0
            # per_local_vector_mmd_loss_temp = torch.FloatTensor().cuda()
            for i in range(len(feature_vector_list)-1):
                for j in range(len(feature_vector_list)-1):
                    if i == j:
                        continue
                    per_local_vector_mmd_loss_temp = mmd(feature_vector_list[i], feature_vector_list[j])
                    per_local_vector_mmd_loss += per_local_vector_mmd_loss_temp
                    loss_mmd_local_vector.add(per_local_vector_mmd_loss_temp.item())
            # ===================== end ==============================#

            # ============= overall loss ================ #
            # loss = ide_loss_local + ide_loss_supply + triplet_loss_supply - df_config.alpha * pre_mmd_loss - df_config.beta * per_local_vector_mmd_loss
            # print(ide_loss_local)
            # print(pre_mmd_loss)
            # print(per_local_vector_mmd_loss)
            # MARKET 8,400
            #DUKE 12,520
            # if (len(trainloader) == batch_idx + 8) or (batch_idx == 400):
            if len(trainloader) == (batch_idx + 8):
                print('updata mmd ',end='')
                print('='*30,'>>')

                # loss = ide_loss_local - df_config.alpha * pre_mmd_loss - df_config.beta * per_local_vector_mmd_loss
                loss = ide_loss_local + pe_loss - 0 * pre_mmd_loss - 0 * per_local_vector_mmd_loss
                # loss = ide_loss_local - df_config.alpha * pre_mmd_loss - 0 * per_local_vector_mmd_loss

            else:

                loss = ide_loss_local + pe_loss
                # loss = ide_loss_local
            # loss = ide_loss_local - df_config.alpha *  pre_mmd_loss - df_config.beta * per_local_vector_mmd_loss

            #====== ============== end ======================#


            #===============statistic======================#
            # print('='*30)
            # print(loss.shape)
            # print(loss.item())
            loss_all.add(loss.item())
            loss_ide_local.add(ide_loss_local.item())
            loss_G2_mse.add(mse_loss.item())
            G2_beta.add(beta.item())
            loss_pe.add(pe_loss.item())
            loss_mmd_pre.add(pre_mmd_loss.item())
            #=================== end =======================#



           # ============ optimize ===============#
            optimizer_G.zero_grad()
            optimizer_G2.zero_grad()
            optimizer_C1.zero_grad()
            # optimizer_C2.zero_grad()

            loss.backward()

            optimizer_G.step()
            optimizer_G2.step()
            optimizer_C1.step()
            # optimizer_C2.step()
           #================ END =====================#





            if batch_idx % df_config.print_freq == 0:
                # viz.plot('--mean of train loss , interval is 20 batchs ', loss_meter.value()[0])
                viz.plot('-- all loss --', loss_all.value()[0])
                viz.plot('-- loss_ide_local --', loss_ide_local.value()[0])
                viz.plot('-- loss_G2_mse --', loss_G2_mse.value()[0])
                viz.plot('-- loss_pe_loss --', loss_pe.value()[0])
                viz.plot('-- G2_beta --', G2_beta.value()[0])
                viz.plot('-- loss_mmd_pre --', loss_mmd_pre.value()[0])
                viz.plot('-- loss_mmd_local_vector --', loss_mmd_local_vector.value()[0])
                print('Epoch: [{0}][{1}/{2}]\t'  'Loss: {3} )\t'
                      .format(
                    epoch + 1, batch_idx + 1, len(trainloader),loss_all.value()[0]))
                # print('id-loss is {}'.format(loss_id.value()[0]))
                # print('mmd loss is {}'.format(loss_mmd.value()[0]))



            # del loss cache
            del  score_maps_temp, score_maps_adjust, beta, sum_counter_list, min_idx, max_idx
            # del score_maps_temp, beta, sum_counter_list, min_idx, max_idx
            # del loss, ide_loss, mmd_loss
        # save weight
        if epoch % df_config.save_freq == 0:
            model_name = str(epoch) + '_' + str(loss_all.value()[0]) + '.pth'
            model_path = os.path.join(df_config.model_path, 'KeypointModel')
            mkdir_if_missing(model_path)
            model_save_path = os.path.join(model_path,model_name)
            torch.save({
                'epoch': epoch,
                'G1_state_dict': G1.state_dict(),
                'G2_state_dict': G2.state_dict(),
                'anchor_state_dict': anchor_net.state_dict(),
                'merger_layer': merger_layer.state_dict(),

            }, model_save_path)
            # torch.save(G1.state_dict(), model_save_path)

        # cal time of epoch
        spend_time_per_epoch = round((time.time() - start_time_per_epoch)/60.0,3)
        spend_all_time += spend_time_per_epoch


        # del cache
        del spend_time_per_epoch, per_local_vector_mmd_loss, feature_maps

        # update lr
        if df_config.step_size > 0:
            lr_scheduler_G.step()
            lr_scheduler_C1.step()
            lr_scheduler_G2.step()

        if (epoch + 1) % df_config.eval_freq == 0 or (epoch + 1) == df_config.max_epoch:

            # test
            print("======== > Test")
            # rank1 = test(model, queryloader, galleryloader, use_gpu, viz=viz)
            # rank1,kkk = testwithVer2(config=df_config, G1=G1, G2=G2, scoremap_computer=scoremap_computer, anchor_net=anchor_net, merger_layer=merger_layer, queryloader=queryloader, galleryloader=galleryloader, bned_feature_vector_list= bned_feature_vector_list, use_gpu=use_gpu, viz=viz)
            # print('-^-'*30)
            # print(rank1)
            # print(kkk)
            # rank1 = new_test(config=df_config, G1=G1, anchor_net=anchor_net, scoremap_computer=scoremap_computer, merger_layer=merger_layer, queryloader=queryloader, galleryloader=galleryloader, use_gpu=use_gpu, viz=viz)
            rank1 = new_test_with_pe(config=df_config, G1=G1, G2=G2, anchor_net=anchor_net, scoremap_computer=scoremap_computer, merger_layer=merger_layer, queryloader=queryloader, galleryloader=galleryloader, use_gpu=use_gpu, viz=viz)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))


    print('################################################')
    print('train model spended--{} Hours'.format(round(spend_all_time/60.0,3)))
    print('################################################')



def new_test(config, G1, scoremap_computer, anchor_net, merger_layer, queryloader, galleryloader, use_gpu, viz):


    G1.eval()
    G2.eval()
    scoremap_computer.eval()

    with torch.no_grad():
        # === = = = = = = load query =  = = = == #
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            if use_gpu:
                # imgs = imgs
                imgs = imgs.cuda()

            #query_features
            features = extract_per_feature_proxy(imgs=imgs, G1=G1,  scoremap_computer= scoremap_computer,anchor_net=anchor_net,merger_layer= merger_layer,config= config)
            # features = model(imgs)

            # features = features.data.cpu()
            qf.append(features)

            #person id   and camera id
            q_pids.extend(pids)
            q_camids.extend(camids)


        qf = torch.cat(qf, 0)

        # query’s person id
        q_pids = np.asarray(q_pids)
        #query's camera id
        q_camids = np.asarray(q_camids)

        # [3368, 12288]
        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        # = = = = = = = load gallery = = = = = = = = #
        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
            if use_gpu: imgs = imgs.cuda()

            # features = model(imgs)
            # features = features.data.cpu()
            features = extract_per_feature_proxy(imgs=imgs, G1=G1, scoremap_computer= scoremap_computer, anchor_net=anchor_net, merger_layer= merger_layer,config= config)
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)

        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        # [13115, 12288]
        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
    #=========================================================#
    print("Computing CMC and mAP")
    query_feature = qf.cuda()
    gallery_feature = gf.cuda()

    # print(query_feature.shape)
    CMC = torch.IntTensor(len(g_pids)).zero_()
    ap = 0.0
    # print(query_label)
    # evaluate_gpu([3368, 512],(3368,),(3368,),[19731, 512], (19731,), (19731,))
    # evaluate_gpu(tensor, numpy.ndarray, numpy.ndarray, tensor, numpy.ndarray, numpy.ndarray)
    for i in range(len(q_pids)):
        ap_tmp, CMC_tmp = evaluate_gpu(query_feature[i], q_pids[i], q_camids[i], gallery_feature, g_pids,
                                       g_camids)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        # print(i, CMC_tmp[0])

    CMC = CMC.float()
    CMC = CMC / len(q_pids)  # average CMC
    mAP = ap / len(q_pids)

    # VIS PLOT
    viz.plot('mAP', mAP)
    viz.plot(' Rank-1 (0-1.0) ', CMC[0])
    viz.plot(' Rank-5 (0-1.0) ', CMC[4])
    viz.plot(' Rank-10 (0-1.0) ', CMC[9])

    print("Results ----------")
    print('Rank-1: {:.1%}   Rank-5: {:.1%}  Rank-10: {:.1%}  mAP: {:.1%}'.format(CMC[0], CMC[4], CMC[9], mAP))
    print("------------------")

    return CMC[0]



def new_test_with_pe(config, G1, G2, scoremap_computer, anchor_net, merger_layer, queryloader, galleryloader, use_gpu, viz):


    G1.eval()
    # G2.eval()
    scoremap_computer.eval()
    # anchor_net.eval()
    # merger_layer.eval()

    with torch.no_grad():
        # === = = = = = =  load query =  = = = == #
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            if use_gpu:
                # imgs = imgs
                imgs = imgs.cuda()

            #query_features
            features = extract_per_feature_proxy_with_G2(imgs=imgs, G1=G1, G2=G2, scoremap_computer= scoremap_computer,anchor_net=anchor_net,merger_layer= merger_layer,config= config)
            # features = model(imgs)

            # features = features.data.cpu()

            qf.append(features)

            q_pids.extend(pids)
            q_camids.extend(camids)

        qf = torch.cat(qf, 0)

        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        # [3368, 12288]
        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        # = = = = = = = load gallery = = = = = = = = #
        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
            if use_gpu: imgs = imgs.cuda()

            # features = model(imgs)
            # features = features.data.cpu()
            features = extract_per_feature_proxy_with_G2(imgs=imgs, G1=G1, G2=G2,scoremap_computer= scoremap_computer, anchor_net=anchor_net, merger_layer= merger_layer,config= config)
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)

        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        # [13115, 12288]
        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
    #=========================================================#
    print("Computing CMC and mAP")
    query_feature = qf.cuda()
    gallery_feature = gf.cuda()

    #======== msmt17 =========#
    # query_feature = qf.cpu()
    # gallery_feature = gf.cpu()

    # print(query_feature.shape)
    CMC = torch.IntTensor(len(g_pids)).zero_()
    ap = 0.0
    # print(query_label)
    # evaluate_gpu([3368, 512],(3368,),(3368,),[19731, 512], (19731,), (19731,))
    # evaluate_gpu(tensor, numpy.ndarray, numpy.ndarray, tensor, numpy.ndarray, numpy.ndarray)
    for i in range(len(q_pids)):
        ap_tmp, CMC_tmp = evaluate_gpu(query_feature[i], q_pids[i], q_camids[i], gallery_feature, g_pids,
                                       g_camids)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        # print(i, CMC_tmp[0])

    CMC = CMC.float()
    CMC = CMC / len(q_pids)  # average CMC
    mAP = ap / len(q_pids)

    # VIS PLOT
    viz.plot('mAP', mAP)
    viz.plot(' Rank-1 (0-1.0) ', CMC[0])
    viz.plot(' Rank-5 (0-1.0) ', CMC[4])
    viz.plot(' Rank-10 (0-1.0) ', CMC[9])

    print("Results ----------")
    print('Rank-1: {:.1%}   Rank-5: {:.1%}  Rank-10: {:.1%}  mAP: {:.1%}'.format(CMC[0], CMC[4], CMC[9], mAP))
    print("------------------")

    return CMC[0]


if __name__ == '__main__':
    main()


