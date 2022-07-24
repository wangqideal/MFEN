class default_config(object):

    # visdom env
    env = 'didiqi_proxy_keypoint'
    port = 9000

    # model name
    model_G = 'proxyNet'

    gpu_devices = '0'

    # is pretrained
    preTrained =False

    # model path
    load_model_path='checkpoints/model.path'
    model_path = 'checkpoints'

    result_dir = 'results'


    #set: market1501, cuhk03, dukemtmcreid
    dataset = 'market1501'
    # root_dir = 'E:/pytorch/dataset/dataset/Duke4U2net'
    root_dir='/didi_files/data'


    img_dir = 'Img'
    label_dir = 'Gt'
    img_ext = '.jpg'
    label_ext = '.png'

    #input sample size
    in_width = 128
    in_height = 256

    #cnhk3
    split_id = 0
    cuhk03_labeled = False
    cuhk03_classic_split = False  # classic
    use_metric_cuhk03 = False

    max_epoch = 70
    train_batch_size = 16
    test_batch_size = 16

    use_gpu = True  # use Gpu
    num_workers = 4 #gpu nums
    lr =0.0003  # learn rating
    weight_decay = 5e-04
    eps = 1e-08
    gamma = 0.1
    betas =(0.9,0.999)
    seed = 123

    print_freq = 20
    step_size = -1
    save_freq = 10
    eval_freq = 10

    # =========mmd factors=========#
    alpha =0.001  # must change
    beta =0.001  #must change
    # =======ide loss========#
    local_branch_num = 13 # need change

    # ====keypoint====
    norm_scale = 10.0         # need change
    weight_global_feature = 1.0 # need change

    # ======triplet loss======#
    margin = 0.3