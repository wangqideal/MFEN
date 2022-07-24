import torch
import os
import time
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torchvision.transforms import transforms as T
from torchvision import models
from torch.utils.data import DataLoader, random_split
from data import ImageDataset
import torch.nn.functional as F

from data import init_img_dataset,get_dataset_name
from torchnet.meter import AverageValueMeter, ConfusionMeter
from torch.backends import cudnn
from config_proxy import default_config
from utils import evaluate, Visualier, mkdir_if_missing, mmd, extract_per_feature, evaluate_gpu
from models import init_model,Predictor


df_config = default_config()
torch.manual_seed(df_config.seed)

device = None
use_gpu =False


# distance of F1 and F2
def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))

def main():
    global device, use_gpu
    os.environ['CUDA_VISABLE_DEVICES'] = default_config.gpu_devices
    use_gpu = torch.cuda.is_available()
    if use_gpu and default_config.use_gpu:
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

    # 是否使用锁页内存
    pin_memory = True if use_gpu else False


    #============== 加载数据 ===============#

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

    # 从disk 加载数据列表
    print("Initializing dataset {}".format(df_config.dataset))
    dataset = init_img_dataset(name=df_config.dataset, root_dir=df_config.root_dir,img_ext=df_config.img_ext,
                     split_id=df_config.split_id, cuhk03_labeled=df_config.cuhk03_labeled, cuhk03_classic_split=df_config.cuhk03_classic_split,
                     )
    #训练集长度
    len_all_train_data = len(dataset.train)

    #分割验证集val 与 训练集train  2：8
    # train_data , val_data = random_split(dataset.train,[round(len_all_train_data*0.9),round(len_all_train_data*0.1)])
    print('## = = = = = = = 加载 数据中 = = = = = = = = = ##')
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

    #加载模型
    print('## = = = = = = = 加载 网络模型 = = = = = = = = = ##')
    print("Initializing model")

    weight_path = './weights/epoch_950_netWeights.pth'
    restNet50 = models.resnet50(pretrained=True)
    pre_model = models.resnet50()
    pre_model.load_state_dict(torch.load(weight_path))

    G = init_model(name=df_config.model_G, backbone1 = pre_model, backbone2 = restNet50 )
    C1 = Predictor(in_dim=2048*2,out_dim=751)
    C2 = Predictor(in_dim=2048*2,out_dim=751)
    #使用 Gpu
    G.to(device)
    C1.to(device)
    C2.to(device)

    #预加载模型
    if df_config.preTrained:
        G.load_state_dict(torch.load(df_config.load_model_path))

    lr = df_config.lr
    #定义损失函数
    criterion = nn.CrossEntropyLoss()
    #定义优化器
    optimizer_G =  torch.optim.Adam(G.parameters(), lr=lr, weight_decay=df_config.weight_decay)
    optimizer_C1 = torch.optim.Adam(C1.parameters(), lr=lr, weight_decay=df_config.weight_decay)
    optimizer_C2 = torch.optim.Adam(C2.parameters(), lr=lr, weight_decay=df_config.weight_decay)


    #定义梯度调整
    lr_scheduler_G=None
    lr_scheduler_C1 = None
    lr_scheduler_C2 = None
    if df_config.step_size>0:
        lr_scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_G, step_size=df_config.step_size, gamma=df_config.gamma)
        lr_scheduler_C1 = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_C1, step_size=df_config.step_size,
                                                   gamma=df_config.gamma)
        lr_scheduler_C2 = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_C2, step_size=df_config.step_size,
                                                   gamma=df_config.gamma)
    #可视化初始化
    viz=Visualier(env=df_config.env, port=df_config.port)
    #数值统计
    loss_meter = AverageValueMeter()
    loss_step_1 = AverageValueMeter()
    loss_step_2 = AverageValueMeter()
    loss_mmd_step_1 = AverageValueMeter()
    loss_mmd_step_2 = AverageValueMeter()
    loss_dis_step_2 = AverageValueMeter()
    loss_dis_step_3 = AverageValueMeter()



    # ============= 训练 ================#
    print('#======== starting  train =========#')
    best_rank1 = -np.inf
    best_epoch = 0
    spend_all_time = 0
    for epoch in tqdm(range(df_config.max_epoch)):
        # 初始化
        start_time_per_epoch = time.time()

        loss_meter.reset()
        loss_step_1.reset()
        loss_step_2.reset()
        loss_mmd_step_1.reset()
        loss_mmd_step_2.reset()
        loss_dis_step_2.reset()
        loss_dis_step_3.reset()

        # 切换为训练模式
        G.train()
        C1.train()
        C2.train()

        for batch_idx, (imgs, pids, cids) in enumerate(trainloader):

            if use_gpu:
                imgs, pids = imgs.to(device), pids.to(device)


            # ===================== #
            # step A
            features_global, features_embedding, embedding = G(imgs)
            mmd_loss_step1 = mmd(embedding[:,:,0], embedding[:,:,1])
            loss_mmd_step_1.add(mmd_loss_step1.item())
            output_s1 = C1(features_global)
            output_s2 = C2(features_global)

            loss_s1 = criterion(output_s1, pids)
            loss_s2 = criterion(output_s2, pids)
            loss_s = loss_s1 + loss_s2 - 0.005*mmd_loss_step1


            loss_step_1.add(loss_s.item())

            optimizer_G.zero_grad()
            optimizer_C1.zero_grad()
            optimizer_C2.zero_grad()

            loss_s.backward()
            optimizer_G.step()
            optimizer_C1.step()
            optimizer_C2.step()

            # ===================== #
            # step B
            # 训练 分类器

            # 提取  source samples features
            features_global, features_embedding, embedding = G(imgs)
            mmd_loss_step2 = mmd(embedding[:, :, 0], embedding[:, :, 1])
            loss_mmd_step_2.add(mmd_loss_step2.item())
            output_s1 = C1(features_global)
            output_s2 = C2(features_global)

            output_t1 = C1(features_embedding)
            output_t2 = C2(features_embedding)

            # 提取  target samples features
            # feat_t = self.G(img_t)
            # output_t1 = self.C1(feat_t)
            # output_t2 = self.C2(feat_t)

            loss_s1 = criterion(output_s1, pids)
            loss_s2 = criterion(output_s2, pids)

            loss_s = loss_s1 + loss_s2

            loss_dis = discrepancy(output_t1, output_t2)
            loss = loss_s - loss_dis - 0.005*mmd_loss_step2

            loss_dis_step_2.add(loss_dis.item())
            loss_step_2.add(loss.item())

            optimizer_G.zero_grad()
            optimizer_C1.zero_grad()
            optimizer_C2.zero_grad()

            loss.backward()

            # 只更新 F1，F2
            optimizer_C1.step()
            optimizer_C2.step()



            # ===================== #
            #训练 特征提取器
            # step C
            # 更新 G 几次
            for i in range(df_config.num_k):
                #
                features_global, features_embedding, embedding = G(imgs)
                output_t1 = C1(features_global)
                output_t2 = C2(features_global)
                loss_dis = discrepancy(output_t1, output_t2)
                loss_dis_step_3.add(loss_dis.item())
                optimizer_G.zero_grad()
                optimizer_C1.zero_grad()
                optimizer_C2.zero_grad()

                loss_dis.backward()

                # 只更新G
                optimizer_G.step()


            # avg_ide_loss = 0
            # avg_logits = 0
            # for i in range(20):
            #     logits_i = logits[i]
            #     avg_logits += 1.0 / float(20) * logits_i
            #     ide_loss_i = criterion(logits_i, pids)
            #     avg_ide_loss += 1.0 / float(20) * ide_loss_i

            #ide_loss : avg_ide_loss , avg_logits
            # ide_loss = avg_ide_loss

            # loss = ide_loss - 0.005*mmd_loss
            # loss = criterion(logits, pids)
            loss_step_1.reset()
            loss_step_2.reset()
            loss_mmd_step_1.reset()
            loss_mmd_step_2.reset()
            loss_dis_step_2.reset()
            loss_dis_step_3.reset()

            if batch_idx % df_config.print_freq == 0:
                # viz.plot('--mean of train loss , interval is 20 batchs ', loss_meter.value()[0])
                viz.plot('-- loss_step_1 --', loss_step_1.value()[0])
                viz.plot('-- loss_step_2 --', loss_step_2.value()[0])
                viz.plot('-- loss_mmd_step_1 --', loss_mmd_step_1.value()[0])
                viz.plot('-- loss_mmd_step_2 --', loss_mmd_step_2.value()[0])

                viz.plot('-- loss_dis_step_2 --', loss_dis_step_2.value()[0])
                viz.plot('-- loss_dis_step_3 --', loss_dis_step_3.value()[0])
                print('Epoch: [{0}][{1}/{2}]\t'                     
                      .format(
                    epoch + 1, batch_idx + 1, len(trainloader)))
                # print('id-loss is {}'.format(loss_id.value()[0]))
                # print('mmd loss is {}'.format(loss_mmd.value()[0]))



            # 删除loss的缓存
            # del loss, ide_loss, mmd_loss
        # 每epoch保存一次
        # 保存模型
        # if epoch % df_config.save_freq == 0:
        #     model_name = str(epoch) + '_' + str(loss_meter.value()) + '.pth'
        #     model_path = os.path.join(df_config.model_path, model.model_name)
        #     mkdir_if_missing(model_path)
        #     model_save_path = os.path.join(model_path,model_name)
        #     torch.save(model.state_dict(), model_save_path)

        # 计算每个epoch 需要多长时间
        spend_time_per_epoch = round(time.time() - start_time_per_epoch,3)
        spend_all_time += spend_time_per_epoch


        #每个 epoch loging
        # viz.log(
        #     "epoch:{epoch},lr:{lr},loss:{loss},##--epoch:{epoch}--spend time：{spend_time}"
        #         .format(
        #         epoch=epoch,
        #         spend_time=spend_time_per_epoch,
        #         loss=loss_meter.value()[0],
        #         lr=lr)
        # )

        # 删除缓存
        del spend_time_per_epoch

        #调用scheduler，step，当 调用总次数 == 参数step_size 即更新学习率
        if df_config.step_size > 0:
            lr_scheduler_G.step()
            lr_scheduler_C1.step()
            lr_scheduler_C2.step()

        if (epoch + 1) % df_config.eval_freq == 0 or (epoch + 1) == df_config.max_epoch:

            # 测试
            print("======== > Test")
            # 测试
            # rank1 = test(model, queryloader, galleryloader, use_gpu, viz=viz)
            rank1 = new_test(G, queryloader, galleryloader, use_gpu, viz=viz)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))


    print('################################################')
    print('train model spended--{} Mins'.format(round(spend_all_time/60.0,3)))
    print('################################################')






#=========验证===========#
def val(net,dataloader):

    #把模型设为验证模式
    net.eval()

    confusion_meter_val = ConfusionMeter(7)


    for batch_idx, data in enumerate(dataloader):
        img, label = data['image'], data['label']
        if use_gpu:
            img, label = img.to(device), label.to(device)

        logits = net(img)
        # logits => torch.Size([4, 7])
        # label=> [4]
        confusion_meter_val.add(logits.data,label.data)


    #恢复为训练模式
    net.trian()
    cm_value=confusion_meter_val.value()
    sum_temp = 0
    for i in range(7):
        sum_temp += cm_value[i][i]

    #PA 像素准确率 预测类别正确的像素数占总像素数的比例
    #对角线元素之和 / 矩阵所有元素之和
    #PA = (TP + TN) / (TP + TN + FP + FN)
    accuracy=(sum_temp)/(cm_value.sum())*100.
    return accuracy

def new_test(model, queryloader, galleryloader, use_gpu, viz):


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
            features = extract_per_feature(imgs=imgs,model=model)
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
            features = extract_per_feature(imgs=imgs, model=model)
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

def train_unet():
    global device, use_gpu
    os.environ['CUDA_VISABLE_DEVICES'] = default_config.gpu_devices
    use_gpu = torch.cuda.is_available()
    if use_gpu and default_config.use_gpu:
        use_gpu = True
    else:
        use_gpu = False
    print('## = = = = = = = = = = = = = = = = ##')

    if use_gpu:
        print("INFO  : % - - - We will use GPU to run - - - %")
        print("The Host has {} GPUs".format(torch.cuda.device_count()))
        print("Currently using GPU {}".format(df_config.gpu_devices))
        # cudnn 加速
        cudnn.benchmark = True
        # 设置随机数种子，保证每次初始化的结果相同
        torch.cuda.manual_seed_all(df_config.seed)
        # 设置 使用哪块 Gpu
        device = torch.device('cuda', int(df_config.gpu_devices))
    else:
        # pass
        print("Currently using CPU (GPU is highly recommended)")

    # 是否使用锁页内存
    pin_memory = True if use_gpu else False

    # ============== 加载数据 ===============#

    transform_train = T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.420,0.402,0.398],std=[0.204,0.202,0.198])
    ])


    # 从disk 加载数据列表
    print("Initializing dataset {}".format(df_config.dataset))
    dataset = init_img_dataset(name=df_config.dataset, root_dir=df_config.root_dir, img_ext=df_config.img_ext,
                               split_id=df_config.split_id, cuhk03_labeled=df_config.cuhk03_labeled,
                               cuhk03_classic_split=df_config.cuhk03_classic_split,
                               )
    # 训练集长度
    len_all_train_data = len(dataset.train)

    # 分割验证集val 与 训练集train  2：8
    # train_data , val_data = random_split(dataset.train,[round(len_all_train_data*0.9),round(len_all_train_data*0.1)])
    print('## = = = = = = = 加载 数据中 = = = = = = = = = ##')
    print('------------------num of train dataset is ：{}'.format(len_all_train_data))

    trainloader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train, is_transform=True),
        batch_size=df_config.train_batch_size, shuffle=True, num_workers=df_config.num_workers,
        pin_memory=pin_memory, drop_last=True,
    )
    #加载模型
    print('## = = = = = = = 加载 网络模型 = = = = = = = = = ##')
    print("Initializing model: {}".format('unet'))
    model = models.init_model(name='unet_didi',in_ch=3,out_ch=3)
    #使用 Gpu
    model = model.to(device)

    #预加载模型
    if df_config.preTrained:
        model.load_state_dict(torch.load(df_config.load_model_path))

    lr = df_config.lr
    #定义损失函数
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.KLDivLoss(reduction='none')
    #定义优化器
    optimizer =  torch.optim.Adam(model.parameters(), lr=lr, weight_decay=df_config.weight_decay)

    #定义梯度调整
    lr_scheduler=None
    if df_config.step_size>0:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=df_config.step_size, gamma=df_config.gamma)

    #可视化初始化
    viz=Visualier(df_config.env)
    #数值统计
    loss_meter = AverageValueMeter()


    # ============= 训练 ================#
    print('#======== starting  train =========#')



    spend_all_time = 0
    for epoch in range(df_config.max_epoch):
        # 初始化
        start_time_per_epoch = time.time()
        loss_meter.reset()

        # 切换为训练模式
        model.train()
        for batch_idx, (imgs, pids, cids) in enumerate(trainloader):
            if use_gpu:
                imgs, pids = imgs.to(device), pids.to(device)

            total_list = list()
            for i in range(default_config.train_batch_size):
                img_temp = imgs[i,:,:,:]


                img_temp = torch.unsqueeze(img_temp, dim=0)
                # print('= = = = = = = = = = = = = = = = = =')
                # print(img_temp.shape)
                # print(type(img_temp))
                img_temp_ = T.Resize((188,60))(img_temp)

                total_list.append(img_temp_)
            img_label = torch.cat(total_list, dim=0)
            # print('============')
            # print(imgs_label.shape)
            logits = model(imgs)
            # logits => torch.Size([4, 7])
            # label=> [4]




            loss = criterion(torch.log(logits), torch.sigmoid(img_label))

            # ========#
            # 清空梯度
            optimizer.zero_grad()
            # 梯度计算
            total_loss = loss.sum()
            total_loss.backward()
            # 更新参数
            optimizer.step()

            # 更新统计指标以及可视化
            loss_meter.add(total_loss.item())

            viz.plot('--train loss --', total_loss.item())
            if batch_idx % df_config.print_freq == df_config.print_freq - 1:
                viz.plot('--mean of train loss , interval is 20 batchs ', loss_meter.value()[0])

                print('Epoch: [{0}][{1}/{2}]\t'                     
                      'Loss: {3} )\t'.format(
                    epoch + 1, batch_idx + 1, len(trainloader), loss_meter.value()[0]))



            # 删除loss的缓存
            del loss
        # 每epoch保存一次
        # 保存模型
        model_name = str(epoch) + '_' + str(loss_meter.value()) + '.pth'
        model_path = os.path.join(df_config.model_path, model.model_name)
        mkdir_if_missing(model_path)
        model_save_path = os.path.join(model_path,model_name)
        torch.save(model.state_dict(), model_save_path)

        # 计算每个epoch 需要多长时间
        spend_time_per_epoch = round(time.time() - start_time_per_epoch)
        spend_all_time += spend_time_per_epoch


        #每个 epoch loging
        viz.log(
            "epoch:{epoch},lr:{lr},loss:{loss},##--epoch:{epoch}--spend time：{spend_time}"
                .format(
                epoch=epoch,
                spend_time=spend_time_per_epoch,
                loss=loss_meter.value()[0],
                lr=lr)
        )

        # 删除缓存
        del spend_time_per_epoch

        #调用scheduler，step，当 调用总次数 == 参数step_size 即更新学习率
        if df_config.step_size > 0: lr_scheduler.step()


    print('################################################')
    print('train model spended--{} Mins'.format(spend_all_time//60.0))
    print('################################################')

if __name__ == '__main__':
    # train_unet()
    main()


