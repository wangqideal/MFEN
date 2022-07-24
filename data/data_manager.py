import glob
import os
import re

import h5py
from imageio import imwrite
from scipy.io import loadmat

from utils import mkdir_if_missing, read_json, write_json


#====  加载数据集 =======#

class Market1501(object):
    """
    Market1501

    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'Market1501'
    def __init__(self,root_dir,img_ext, **kwargs):
        self.dataset_name = 'Market1501'
        self.dataset_dir = os.path.join(root_dir,self.dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir,'bounding_box_train')
        self.query_dir = os.path.join(self.dataset_dir, 'query')
        self.gallery_dir = os.path.join(self.dataset_dir, 'bounding_box_test')
        self.img_ext = img_ext
        #检查路径是否正确
        self._check_path()

        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False)
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs


        print("=> Market1501 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    #检查路径是否存在
    def _check_path(self):
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not os.path.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not os.path.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not os.path.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    #relabel 将
    def _process_dir(self,dir_path,relabel=False):
        img_path_list = glob.glob(os.path.join(dir_path,'*' + self.img_ext))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()

        for img_path  in img_path_list:
          p_id, c_id = map(int,pattern.search(img_path).groups())
          if p_id == -1 : continue
          pid_container.add(p_id)

        #pid == 原ID；label == 0-750 新ID  方便后面使用新ID进行构建网络，即网络中使用0-750作为标签
        pid2label = {pid : label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_path_list:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))
            #数据结构--》（path , person-id, camera-id）

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return  dataset, num_pids, num_imgs

class Market1501_Partial(object):
    """
    Market1501

    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'market1501_partial'

    def __init__(self, root_dir='data', **kwargs):
        self.dataset_dir = os.path.join(root_dir, self.dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = os.path.join(self.dataset_dir, 'query')
        self.gallery_dir = os.path.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False)
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> Market1501 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not os.path.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not os.path.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not os.path.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(os.path.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs

# class CUHK03(object):
#     """
#     CUHK03
#
#     Reference:
#     Li et al. DeepReID: Deep Filter Pairing Neural Network for Person Re-identification. CVPR 2014.
#
#     URL: http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html#!
#
#     Dataset statistics:
#     # identities: 1360
#     # images: 13164
#     # cameras: 6
#     # splits: 20 (classic)
#
#     Args:
#         split_id (int): split index (default: 0)
#         cuhk03_labeled (bool): whether to load labeled images; if false, detected images are loaded (default: False)
#     """
#     dataset_dir = 'CUHK03'
#
#     def __init__(self, root_dir, split_id=0, cuhk03_labeled=False, cuhk03_classic_split=False, **kwargs):
#         self.dataset_name = 'CUHK03'
#         self.dataset_dir = os.path.join(root_dir, self.dataset_dir)
#         self.data_dir = os.path.join(self.dataset_dir, 'cuhk03_release')
#         # dataset_dir +  data_dir + cuhk-03.mat
#         # 数据集文件夹位置 + cuhk03_release + cuhk-03.mat
#         self.raw_mat_path = os.path.join(self.data_dir, 'cuhk-03.mat')
#
#         #经典分割数据集方式
#         #创建目录
#         #detected
#         self.imgs_detected_dir = os.path.join(self.dataset_dir, 'images_detected')
#         #label
#         self.imgs_labeled_dir = os.path.join(self.dataset_dir, 'images_labeled')
#
#         #创建文件
#         #classic_detected
#         self.split_classic_det_json_path = os.path.join(self.dataset_dir, 'splits_classic_detected.json')
#         #classic_label
#         self.split_classic_lab_json_path = os.path.join(self.dataset_dir, 'splits_classic_labeled.json')
#
#         #new_detected
#         self.split_new_det_json_path = os.path.join(self.dataset_dir, 'splits_new_detected.json')
#         #new_label
#         self.split_new_lab_json_path = os.path.join(self.dataset_dir, 'splits_new_labeled.json')
#
#         #第二种 新协议，新的分割数据集方式
#         self.split_new_det_mat_path = os.path.join(self.dataset_dir, 'cuhk03_new_protocol_config_detected.mat')
#         self.split_new_lab_mat_path = os.path.join(self.dataset_dir, 'cuhk03_new_protocol_config_labeled.mat')
#
#         #检查路径
#         self._check_before_run()
#         #处理数据
#         self._preprocess()
#
#         #选择label or detected
#         if cuhk03_labeled:
#             image_type = 'labeled'
#             # classic_split  or new_split
#             split_path = self.split_classic_lab_json_path if cuhk03_classic_split else self.split_new_lab_json_path
#         else:
#             image_type = 'detected'
#             # classic_split  or new_split
#             split_path = self.split_classic_det_json_path if cuhk03_classic_split else self.split_new_det_json_path
#
#
#         splits = read_json(split_path)
#         assert split_id < len(splits), "Condition split_id ({}) < len(splits) ({}) is false".format(split_id,
#                                                                                                     len(splits))
#         split = splits[split_id]
#         print("Split index = {}".format(split_id))
#
#         train = split['train']
#         query = split['query']
#         gallery = split['gallery']
#
#         num_train_pids = split['num_train_pids']
#         num_query_pids = split['num_query_pids']
#         num_gallery_pids = split['num_gallery_pids']
#         num_total_pids = num_train_pids + num_query_pids
#
#         num_train_imgs = split['num_train_imgs']
#         num_query_imgs = split['num_query_imgs']
#         num_gallery_imgs = split['num_gallery_imgs']
#         num_total_imgs = num_train_imgs + num_query_imgs
#
#         print("=> CUHK03 ({}) loaded".format(image_type))
#         print("Dataset statistics:")
#         print("  ------------------------------")
#         print("  subset   | # ids | # images")
#         print("  ------------------------------")
#         print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
#         print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
#         print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
#         print("  ------------------------------")
#         print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
#         print("  ------------------------------")
#
#         self.train = train
#         self.query = query
#         self.gallery = gallery
#
#         self.num_train_pids = num_train_pids
#         self.num_query_pids = num_query_pids
#         self.num_gallery_pids = num_gallery_pids
#
#     def _check_before_run(self):
#         """Check if all files are available before going deeper"""
#         if not os.path.exists(self.dataset_dir):
#             raise RuntimeError("'{}' is not available".format(self.dataset_dir))
#         if not os.path.exists(self.data_dir):
#             raise RuntimeError("'{}' is not available".format(self.data_dir))
#         if not os.path.exists(self.raw_mat_path):
#             raise RuntimeError("'{}' is not available".format(self.raw_mat_path))
#         if not os.path.exists(self.split_new_det_mat_path):
#             raise RuntimeError("'{}' is not available".format(self.split_new_det_mat_path))
#         if not os.path.exists(self.split_new_lab_mat_path):
#             raise RuntimeError("'{}' is not available".format(self.split_new_lab_mat_path))
#
#     def _preprocess(self):
#         """
#         This function is a bit complex and ugly, what it does is
#         1. Extract data from cuhk-03.mat and save as png images.
#         2. Create 20 classic splits. (Li et al. CVPR'14)
#         3. Create new split. (Zhong et al. CVPR'17)
#         """
#
#         #检查是否 已存在分割好的数据集，如有则退出
#         print(
#             "Note: if root path is changed, the previously generated json files need to be re-generated (delete them first)")
#         if os.path.exists(self.imgs_labeled_dir) and \
#                 os.path.exists(self.imgs_detected_dir) and \
#                 os.path.exists(self.split_classic_det_json_path) and \
#                 os.path.exists(self.split_classic_lab_json_path) and \
#                 os.path.exists(self.split_new_det_json_path) and \
#                 os.path.exists(self.split_new_lab_json_path):
#             return
#
#         mkdir_if_missing(self.imgs_detected_dir)
#         mkdir_if_missing(self.imgs_labeled_dir)
#
#         #读取cuhk_realse.mat 文件
#         print("Extract image data from {} and save as png".format(self.raw_mat_path))
#         mat = h5py.File(self.raw_mat_path, 'r')
#
#         #行 ---> 列
#         def _deref(ref):
#             return mat[ref][:].T
#
#         def _process_images(img_refs, campid, pid, save_dir):
#             img_paths = []  # Note: some persons only have images for one view
#             for imgid, img_ref in enumerate(img_refs):
#                 img = _deref(img_ref)
#                 # skip empty cell
#                 if img.size == 0 or img.ndim < 3: continue
#                 # images are saved with the following format, index-1 (ensure uniqueness)
#                 # campid: index of camera pair (1-5)
#                 # pid: index of person in 'campid'-th camera pair
#                 # viewid: index of view, {1, 2}
#                 # imgid: index of image, (1-10)
#                 viewid = 1 if imgid < 5 else 2
#                 img_name = '{:01d}_{:03d}_{:01d}_{:02d}.png'.format(campid + 1, pid + 1, viewid, imgid + 1)
#                 img_path = os.path.join(save_dir, img_name)
#                 # imsave(img_path, img)
#                 imwrite(img_path, img)
#                 img_paths.append(img_path)
#             return img_paths
#
#         def _extract_img(name):
#             print("Processing {} images (extract and save) ...".format(name))
#             meta_data = []
#             imgs_dir = self.imgs_detected_dir if name == 'detected' else self.imgs_labeled_dir
#             for campid, camp_ref in enumerate(mat[name][0]):
#                 camp = _deref(camp_ref)
#                 num_pids = camp.shape[0]
#                 for pid in range(num_pids):
#                     img_paths = _process_images(camp[pid, :], campid, pid, imgs_dir)
#                     assert len(img_paths) > 0, "campid{}-pid{} has no images".format(campid, pid)
#                     meta_data.append((campid + 1, pid + 1, img_paths))
#                 print("done camera pair {} with {} identities".format(campid + 1, num_pids))
#             return meta_data
#
#         meta_detected = _extract_img('detected')
#         meta_labeled = _extract_img('labeled')
#
#         def _extract_classic_split(meta_data, test_split):
#             train, test = [], []
#             num_train_pids, num_test_pids = 0, 0
#             num_train_imgs, num_test_imgs = 0, 0
#             for i, (campid, pid, img_paths) in enumerate(meta_data):
#
#                 if [campid, pid] in test_split:
#                     for img_path in img_paths:
#                         camid = int(os.path.basename(img_path).split('_')[2])
#                         test.append((img_path, num_test_pids, camid))
#                     num_test_pids += 1
#                     num_test_imgs += len(img_paths)
#                 else:
#                     for img_path in img_paths:
#                         camid = int(os.path.basename(img_path).split('_')[2])
#                         train.append((img_path, num_train_pids, camid))
#                     num_train_pids += 1
#                     num_train_imgs += len(img_paths)
#             return train, num_train_pids, num_train_imgs, test, num_test_pids, num_test_imgs
#
#         print("Creating classic splits (# = 20) ...")
#         splits_classic_det, splits_classic_lab = [], []
#         for split_ref in mat['testsets'][0]:
#             test_split = _deref(split_ref).tolist()
#
#             # create split for detected images
#             train, num_train_pids, num_train_imgs, test, num_test_pids, num_test_imgs = \
#                 _extract_classic_split(meta_detected, test_split)
#             splits_classic_det.append({
#                 'train': train, 'query': test, 'gallery': test,
#                 'num_train_pids': num_train_pids, 'num_train_imgs': num_train_imgs,
#                 'num_query_pids': num_test_pids, 'num_query_imgs': num_test_imgs,
#                 'num_gallery_pids': num_test_pids, 'num_gallery_imgs': num_test_imgs,
#             })
#
#             # create split for labeled images
#             train, num_train_pids, num_train_imgs, test, num_test_pids, num_test_imgs = \
#                 _extract_classic_split(meta_labeled, test_split)
#             splits_classic_lab.append({
#                 'train': train, 'query': test, 'gallery': test,
#                 'num_train_pids': num_train_pids, 'num_train_imgs': num_train_imgs,
#                 'num_query_pids': num_test_pids, 'num_query_imgs': num_test_imgs,
#                 'num_gallery_pids': num_test_pids, 'num_gallery_imgs': num_test_imgs,
#             })
#
#         write_json(splits_classic_det, self.split_classic_det_json_path)
#         write_json(splits_classic_lab, self.split_classic_lab_json_path)
#
#         def _extract_set(filelist, pids, pid2label, idxs, img_dir, relabel):
#             tmp_set = []
#             unique_pids = set()
#             for idx in idxs:
#                 img_name = filelist[idx][0]
#                 camid = int(img_name.split('_')[2])
#                 pid = pids[idx]
#                 if relabel: pid = pid2label[pid]
#                 img_path = os.path.join(img_dir, img_name)
#                 tmp_set.append((img_path, int(pid), camid))
#                 unique_pids.add(pid)
#             return tmp_set, len(unique_pids), len(idxs)
#
#         def _extract_new_split(split_dict, img_dir):
#             train_idxs = split_dict['train_idx'].flatten() - 1  # index-0
#             pids = split_dict['labels'].flatten()
#             train_pids = set(pids[train_idxs])
#             pid2label = {pid: label for label, pid in enumerate(train_pids)}
#             query_idxs = split_dict['query_idx'].flatten() - 1
#             gallery_idxs = split_dict['gallery_idx'].flatten() - 1
#             filelist = split_dict['filelist'].flatten()
#             train_info = _extract_set(filelist, pids, pid2label, train_idxs, img_dir, relabel=True)
#             query_info = _extract_set(filelist, pids, pid2label, query_idxs, img_dir, relabel=False)
#             gallery_info = _extract_set(filelist, pids, pid2label, gallery_idxs, img_dir, relabel=False)
#             return train_info, query_info, gallery_info
#
#         print("Creating new splits for detected images (767/700) ...")
#         train_info, query_info, gallery_info = _extract_new_split(
#             loadmat(self.split_new_det_mat_path),
#             self.imgs_detected_dir,
#         )
#         splits = [{
#             'train': train_info[0], 'query': query_info[0], 'gallery': gallery_info[0],
#             'num_train_pids': train_info[1], 'num_train_imgs': train_info[2],
#             'num_query_pids': query_info[1], 'num_query_imgs': query_info[2],
#             'num_gallery_pids': gallery_info[1], 'num_gallery_imgs': gallery_info[2],
#         }]
#         write_json(splits, self.split_new_det_json_path)
#
#         print("Creating new splits for labeled images (767/700) ...")
#         train_info, query_info, gallery_info = _extract_new_split(
#             loadmat(self.split_new_lab_mat_path),
#             self.imgs_labeled_dir,
#         )
#         splits = [{
#             'train': train_info[0], 'query': query_info[0], 'gallery': gallery_info[0],
#             'num_train_pids': train_info[1], 'num_train_imgs': train_info[2],
#             'num_query_pids': query_info[1], 'num_query_imgs': query_info[2],
#             'num_gallery_pids': gallery_info[1], 'num_gallery_imgs': gallery_info[2],
#         }]
#         write_json(splits, self.split_new_lab_json_path)


class CUHK03(object):
    """
    CUHK03

    Reference:
    Li et al. DeepReID: Deep Filter Pairing Neural Network for Person Re-identification. CVPR 2014.

    URL: http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html#!

    Dataset statistics:
    # identities: 1360
    # images: 13164
    # cameras: 6
    # splits: 20 (classic)

    Args:
        split_id (int): split index (default: 0)
        cuhk03_labeled (bool): whether to load labeled images; if false, detected images are loaded (default: False)
    """
    dataset_dir = 'CUHK03'

    def __init__(self, root='data', split_id=0, cuhk03_labeled=False, cuhk03_classic_split=False, **kwargs):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.data_dir = os.path.join(self.dataset_dir, 'cuhk03_release')
        self.raw_mat_path = os.path.join(self.data_dir, 'cuhk-03.mat')

        self.imgs_detected_dir = os.path.join(self.dataset_dir, 'images_detected')
        self.imgs_labeled_dir = os.path.join(self.dataset_dir, 'images_labeled')

        self.split_classic_det_json_path = os.path.join(self.dataset_dir, 'splits_classic_detected.json')
        self.split_classic_lab_json_path = os.path.join(self.dataset_dir, 'splits_classic_labeled.json')

        self.split_new_det_json_path = os.path.join(self.dataset_dir, 'splits_new_detected.json')
        self.split_new_lab_json_path = os.path.join(self.dataset_dir, 'splits_new_labeled.json')

        self.split_new_det_mat_path = os.path.join(self.dataset_dir, 'cuhk03_new_protocol_config_detected.mat')
        self.split_new_lab_mat_path = os.path.join(self.dataset_dir, 'cuhk03_new_protocol_config_labeled.mat')

        self._check_before_run()
        self._preprocess()

        if cuhk03_labeled:
            image_type = 'labeled'
            split_path = self.split_classic_lab_json_path if cuhk03_classic_split else self.split_new_lab_json_path
        else:
            image_type = 'detected'
            split_path = self.split_classic_det_json_path if cuhk03_classic_split else self.split_new_det_json_path

        splits = read_json(split_path)
        assert split_id < len(splits), "Condition split_id ({}) < len(splits) ({}) is false".format(split_id,
                                                                                                    len(splits))
        split = splits[split_id]
        print("Split index = {}".format(split_id))

        train = split['train']
        query = split['query']
        gallery = split['gallery']

        num_train_pids = split['num_train_pids']
        num_query_pids = split['num_query_pids']
        num_gallery_pids = split['num_gallery_pids']
        num_total_pids = num_train_pids + num_query_pids

        num_train_imgs = split['num_train_imgs']
        num_query_imgs = split['num_query_imgs']
        num_gallery_imgs = split['num_gallery_imgs']
        num_total_imgs = num_train_imgs + num_query_imgs

        print("=> CUHK03 ({}) loaded".format(image_type))
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not os.path.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))
        if not os.path.exists(self.raw_mat_path):
            raise RuntimeError("'{}' is not available".format(self.raw_mat_path))
        if not os.path.exists(self.split_new_det_mat_path):
            raise RuntimeError("'{}' is not available".format(self.split_new_det_mat_path))
        if not os.path.exists(self.split_new_lab_mat_path):
            raise RuntimeError("'{}' is not available".format(self.split_new_lab_mat_path))

    def _preprocess(self):
        """
        This function is a bit complex and ugly, what it does is
        1. Extract data from cuhk-03.mat and save as png images.
        2. Create 20 classic splits. (Li et al. CVPR'14)
        3. Create new split. (Zhong et al. CVPR'17)
        """
        print(
            "Note: if root path is changed, the previously generated json files need to be re-generated (delete them first)")
        if os.path.exists(self.imgs_labeled_dir) and \
                os.path.exists(self.imgs_detected_dir) and \
                os.path.exists(self.split_classic_det_json_path) and \
                os.path.exists(self.split_classic_lab_json_path) and \
                os.path.exists(self.split_new_det_json_path) and \
                os.path.exists(self.split_new_lab_json_path):
            return

        mkdir_if_missing(self.imgs_detected_dir)
        mkdir_if_missing(self.imgs_labeled_dir)

        print("Extract image data from {} and save as png".format(self.raw_mat_path))
        mat = h5py.File(self.raw_mat_path, 'r')

        def _deref(ref):
            return mat[ref][:].T

        def _process_images(img_refs, campid, pid, save_dir):
            img_paths = []  # Note: some persons only have images for one view
            for imgid, img_ref in enumerate(img_refs):
                img = _deref(img_ref)
                # skip empty cell
                if img.size == 0 or img.ndim < 3: continue
                # images are saved with the following format, index-1 (ensure uniqueness)
                # campid: index of camera pair (1-5)
                # pid: index of person in 'campid'-th camera pair
                # viewid: index of view, {1, 2}
                # imgid: index of image, (1-10)
                viewid = 1 if imgid < 5 else 2
                img_name = '{:01d}_{:03d}_{:01d}_{:02d}.png'.format(campid + 1, pid + 1, viewid, imgid + 1)
                img_path = os.path.join(save_dir, img_name)
                imsave(img_path, img)
                img_paths.append(img_path)
            return img_paths

        def _extract_img(name):
            print("Processing {} images (extract and save) ...".format(name))
            meta_data = []
            imgs_dir = self.imgs_detected_dir if name == 'detected' else self.imgs_labeled_dir
            for campid, camp_ref in enumerate(mat[name][0]):
                camp = _deref(camp_ref)
                num_pids = camp.shape[0]
                for pid in range(num_pids):
                    img_paths = _process_images(camp[pid, :], campid, pid, imgs_dir)
                    assert len(img_paths) > 0, "campid{}-pid{} has no images".format(campid, pid)
                    meta_data.append((campid + 1, pid + 1, img_paths))
                print("done camera pair {} with {} identities".format(campid + 1, num_pids))
            return meta_data

        meta_detected = _extract_img('detected')
        meta_labeled = _extract_img('labeled')

        def _extract_classic_split(meta_data, test_split):
            train, test = [], []
            num_train_pids, num_test_pids = 0, 0
            num_train_imgs, num_test_imgs = 0, 0
            for i, (campid, pid, img_paths) in enumerate(meta_data):

                if [campid, pid] in test_split:
                    for img_path in img_paths:
                        camid = int(os.path.basename(img_path).split('_')[2])
                        test.append((img_path, num_test_pids, camid))
                    num_test_pids += 1
                    num_test_imgs += len(img_paths)
                else:
                    for img_path in img_paths:
                        camid = int(os.path.basename(img_path).split('_')[2])
                        train.append((img_path, num_train_pids, camid))
                    num_train_pids += 1
                    num_train_imgs += len(img_paths)
            return train, num_train_pids, num_train_imgs, test, num_test_pids, num_test_imgs

        print("Creating classic splits (# = 20) ...")
        splits_classic_det, splits_classic_lab = [], []
        for split_ref in mat['testsets'][0]:
            test_split = _deref(split_ref).tolist()

            # create split for detected images
            train, num_train_pids, num_train_imgs, test, num_test_pids, num_test_imgs = \
                _extract_classic_split(meta_detected, test_split)
            splits_classic_det.append({
                'train': train, 'query': test, 'gallery': test,
                'num_train_pids': num_train_pids, 'num_train_imgs': num_train_imgs,
                'num_query_pids': num_test_pids, 'num_query_imgs': num_test_imgs,
                'num_gallery_pids': num_test_pids, 'num_gallery_imgs': num_test_imgs,
            })

            # create split for labeled images
            train, num_train_pids, num_train_imgs, test, num_test_pids, num_test_imgs = \
                _extract_classic_split(meta_labeled, test_split)
            splits_classic_lab.append({
                'train': train, 'query': test, 'gallery': test,
                'num_train_pids': num_train_pids, 'num_train_imgs': num_train_imgs,
                'num_query_pids': num_test_pids, 'num_query_imgs': num_test_imgs,
                'num_gallery_pids': num_test_pids, 'num_gallery_imgs': num_test_imgs,
            })

        write_json(splits_classic_det, self.split_classic_det_json_path)
        write_json(splits_classic_lab, self.split_classic_lab_json_path)

        def _extract_set(filelist, pids, pid2label, idxs, img_dir, relabel):
            tmp_set = []
            unique_pids = set()
            for idx in idxs:
                img_name = filelist[idx][0]
                camid = int(img_name.split('_')[2])
                pid = pids[idx]
                if relabel: pid = pid2label[pid]
                img_path = os.path.join(img_dir, img_name)
                tmp_set.append((img_path, int(pid), camid))
                unique_pids.add(pid)
            return tmp_set, len(unique_pids), len(idxs)

        def _extract_new_split(split_dict, img_dir):
            train_idxs = split_dict['train_idx'].flatten() - 1  # index-0
            pids = split_dict['labels'].flatten()
            train_pids = set(pids[train_idxs])
            pid2label = {pid: label for label, pid in enumerate(train_pids)}
            query_idxs = split_dict['query_idx'].flatten() - 1
            gallery_idxs = split_dict['gallery_idx'].flatten() - 1
            filelist = split_dict['filelist'].flatten()
            train_info = _extract_set(filelist, pids, pid2label, train_idxs, img_dir, relabel=True)
            query_info = _extract_set(filelist, pids, pid2label, query_idxs, img_dir, relabel=False)
            gallery_info = _extract_set(filelist, pids, pid2label, gallery_idxs, img_dir, relabel=False)
            return train_info, query_info, gallery_info

        print("Creating new splits for detected images (767/700) ...")
        train_info, query_info, gallery_info = _extract_new_split(
            loadmat(self.split_new_det_mat_path),
            self.imgs_detected_dir,
        )
        splits = [{
            'train': train_info[0], 'query': query_info[0], 'gallery': gallery_info[0],
            'num_train_pids': train_info[1], 'num_train_imgs': train_info[2],
            'num_query_pids': query_info[1], 'num_query_imgs': query_info[2],
            'num_gallery_pids': gallery_info[1], 'num_gallery_imgs': gallery_info[2],
        }]
        write_json(splits, self.split_new_det_json_path)

        print("Creating new splits for labeled images (767/700) ...")
        train_info, query_info, gallery_info = _extract_new_split(
            loadmat(self.split_new_lab_mat_path),
            self.imgs_labeled_dir,
        )
        splits = [{
            'train': train_info[0], 'query': query_info[0], 'gallery': gallery_info[0],
            'num_train_pids': train_info[1], 'num_train_imgs': train_info[2],
            'num_query_pids': query_info[1], 'num_query_imgs': query_info[2],
            'num_gallery_pids': gallery_info[1], 'num_gallery_imgs': gallery_info[2],
        }]
        write_json(splits, self.split_new_lab_json_path)

class DukeMTMCreID(object):
    """
    DukeMTMC-reID

    Reference:
    1. Ristani et al. Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking. ECCVW 2016.
    2. Zheng et al. Unlabeled Samples Generated by GAN Improve the Person Re-identification Baseline in vitro. ICCV 2017.

    URL: https://github.com/layumi/DukeMTMC-reID_evaluation

    Dataset statistics:
    # identities: 1404 (train + query)
    # images:16522 (train) + 2228 (query) + 17661 (gallery)
    # cameras: 8
    """
    dataset_dir = 'DukeMTMC-reID'

    def __init__(self, root_dir='data', **kwargs):
        self.dataset_name = 'DukeMTMCreID'
        self.dataset_dir = os.path.join(root_dir, self.dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = os.path.join(self.dataset_dir, 'query')
        self.gallery_dir = os.path.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False)
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> DukeMTMC-reID loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not os.path.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not os.path.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not os.path.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(os.path.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 1 <= camid <= 8
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs



"""Create dataset"""

__img_factory = {
    'market1501': Market1501,
    'cuhk03': CUHK03,
    'dukemtmcreid': DukeMTMCreID,
    # 'msmt17': MSMT17,
}

def get_dataset_name():
    return str(__img_factory.keys())

#初始化dataset
def init_img_dataset(name,**kwargs):
    if name not in __img_factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, __img_factory.keys()))
    return __img_factory[name](**kwargs)
