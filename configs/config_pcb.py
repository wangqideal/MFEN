class config_PCB(object):

    # visdom 环境
    env = 'didiqi_offical_pcb'
    port = 9000

    #====test======#
    use_pcb = True
    use_proxy = False
    # 基础网络
    arch = 'resnet50'
    # 要加载的模型名称
    model = 'pcb'

    gpu_devices = '0'

    # 是否使用已训练的参数
    preTrained =False

    # 加载已存在模型的路径
    load_model_path='checkpoints/model.path'
    model_path = 'checkpoints'

    result_dir = 'results'


    #数据集 market1501, cuhk03, dukemtmcreid
    #使用哪个数据集
    dataset = 'market1501'
    # root_dir = 'E:/pytorch/dataset/dataset/Duke4U2net'
    root_dir='/didi_files/data'
    # 数据集路
    # root_dir ='E:/pytorch/dataset/test_edu'

    img_dir = 'Img'
    label_dir = 'Gt'
    # 文件后缀
    img_ext = '.jpg'
    label_ext = '.png'

    #输入图像大小
    in_width = 128
    in_height = 256

    #cnhk3数据集 配置
    split_id = 0
    cuhk03_labeled = False
    cuhk03_classic_split = False  #使用经典分割数据集方法
    use_metric_cuhk03 = False   #对cuhk3进行评价？

    max_epoch = 70
    train_batch_size = 32
    test_batch_size = 32

    use_gpu = True  #是否使用Gpu
    num_workers = 4 #dataloader 使用cpu个数
    lr =0.0002  #学习率
    weight_decay = 5e-04  #学习率更新参数
    eps = 1e-08
    gamma = 0.1  # 学习率更新 参数
    betas =(0.9,0.999)
    seed = 123  #随即种子

    print_freq = 20  #信息打印频率
    step_size = 20   #学习率更新频率
    save_freq = 10
    eval_freq = 5



