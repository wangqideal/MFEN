import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchnet.meter import AverageValueMeter, ConfusionMeter
from torchvision.transforms import transforms as T
from tqdm import tqdm

from configs import config_PROXY
from data import ImageDataset
from data import init_img_dataset
from models import init_model, BNClassifiers
from utils import Visualier, mkdir_if_missing, extract_per_feature, evaluate_gpu

df_config = config_PROXY()
torch.manual_seed(df_config.seed)

device = None
use_gpu =False


# F1 F2 距离
def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))

def compute_ide_loss(ide_creiteron, score_list, pids, banch=3):
    loss_all = 0
    for i, score_i in enumerate(score_list):
        loss_i = ide_creiteron(score_i, pids)
        loss_all += 1.0 / float(banch) * loss_i
    return loss_all

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


    #============== load data ===============#

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
    #len of train set
    len_all_train_data = len(dataset.train)

    #val set : train set  2：8
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


    print('## = = = = = = = loading model = = = = = = = = = ##')
    print("Initializing model")

    net = init_model(df_config.model, class_num = 751)
    bnclassifers = BNClassifiers(2048,751,3)

    #use Gpu
    net.to(device)
    bnclassifers.to(device)

    #load pretrained weight
    if df_config.preTrained:
        net.load_state_dict(torch.load(df_config.load_model_path))
    lr = df_config.lr
    # loss fun
    criterion = nn.CrossEntropyLoss()
    #optim
    optimizer =  torch.optim.Adam(net.parameters(), lr=lr, weight_decay=df_config.weight_decay)
    optimizer_C1  = torch.optim.Adam(bnclassifers.parameters(), lr=lr, weight_decay=df_config.weight_decay)

    lr_scheduler=None
    lr_scheduler_C1=None
    if df_config.step_size>0:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=df_config.step_size, gamma=df_config.gamma)
        lr_scheduler_C1 = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_C1, step_size=df_config.step_size, gamma=df_config.gamma)

    # init visdom
    viz=Visualier(env=df_config.env, port=df_config.port)
    # statistic
    loss_all = AverageValueMeter()
    loss_avg = AverageValueMeter()




    # ============= train ================#
    print('#======== starting  train =========#')
    best_rank1 = -np.inf
    best_epoch = 0
    spend_all_time = 0
    for epoch in tqdm(range(df_config.max_epoch)):
        # init
        start_time_per_epoch = time.time()

        loss_all.reset()
        loss_avg.reset()

        # switch mode
        net.train()

        for batch_idx, (imgs, pids, cids) in enumerate(trainloader):

            if use_gpu:
                imgs, pids = imgs.to(device), pids.to(device)


            # ===================== #

            logits, embedding = net(imgs)
            features_list = list()
            for i in range(logits.size(3)):
                features_list.append(logits[:,:,i])
            cls_list = bnclassifers(features_list)
            ide_loss = compute_ide_loss(criterion,cls_list,pids,banch=3)


            #============================#
            loss_all.add(ide_loss.item())

            optimizer.zero_grad()
            optimizer_C1.zero_grad()

            ide_loss.backward()

            optimizer.step()
            optimizer_C1.step()

            if batch_idx % df_config.print_freq == 0:
                # viz.plot('--mean of train loss , interval is 20 batchs ', loss_meter.value()[0])
                viz.plot('-- loss_all --', loss_all.value()[0])
                # viz.plot('-- loss_avg --', loss_avg.value()[0])

                print('Epoch: [{0}][{1}/{2}]\t' 'Loss : {3}\t'
                      .format(
                    epoch + 1, batch_idx + 1, len(trainloader),loss_all.value()[0]))



            # de loss cache
            del  ide_loss

        # save model
        if epoch % df_config.save_freq == 0:
            model_name = str(epoch) + '_' + str(loss_all.value()[0]) + '.pth'
            model_path = os.path.join(df_config.model_path, net.model_name)
            mkdir_if_missing(model_path)
            model_save_path = os.path.join(model_path,model_name)
            torch.save(net.state_dict(), model_save_path)

        # cal time
        spend_time_per_epoch = round((time.time() - start_time_per_epoch)/60.0,3)
        spend_all_time += spend_time_per_epoch




        # del cache
        del spend_time_per_epoch

        # update lr
        if df_config.step_size > 0:
            lr_scheduler.step()
            lr_scheduler_C1.step()


        if (epoch + 1) % df_config.eval_freq == 0 or (epoch + 1) == df_config.max_epoch:

            # test
            print("======== > Test")
            # rank1 = test(model, queryloader, galleryloader, use_gpu, viz=viz)
            rank1 = new_test(net, queryloader, galleryloader, use_gpu, viz=viz, config = df_config)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))


    print('################################################')
    print('train model spended--{} Mins'.format(round(spend_all_time/60.0,3)))
    print('################################################')







def new_test(model, queryloader, galleryloader, use_gpu, viz, config):


    model.eval()

    with torch.no_grad():
        # === = = = = = = 加载所有query =  = = = == #
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            if use_gpu:
                # imgs = imgs
                imgs = imgs.cuda()

            #query_features
            #[batch_size,所有网络的输出]
            features = extract_per_feature(imgs=imgs,model=model,config=config)
            # features = model(imgs)

            # 所有  query 的feature(未经过fc层 之前的map)
            # features = features.data.cpu()

            #每个预测结果
            qf.append(features)

            #每个预测的person id  和 camera id
            q_pids.extend(pids)
            q_camids.extend(camids)

        #每个批次的预测结果
        #将dim = 0, 全部拆分
        # [3368,12288]， 3368 == len(query) , 12288 == 每个query中图片的特征（在fc分类层之前的map）
        qf = torch.cat(qf, 0)

        #所有query 的 person id
        q_pids = np.asarray(q_pids)
        #所有 query 的 camera id
        q_camids = np.asarray(q_camids)

        # [3368, 12288]
        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        # = = = = = = = 加载所有 gallery = = = = = = = = #
        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
            if use_gpu: imgs = imgs.cuda()

            #所有  gallery 的feature(未经过fc层 之前的map)
            # features = model(imgs)
            # features = features.data.cpu()
            features = extract_per_feature(imgs=imgs, model=model,config=config)
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)

        #[13115, 12288] , 13115 所有gallery, 12228每个gallery图片的 特征
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



if __name__ == '__main__':
    # train_unet()
    main()


