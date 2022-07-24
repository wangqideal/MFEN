import os

from PIL import Image
from torch.utils.data import Dataset


def read_image(img_path):
    img = None
    got_img = False
    if not os.path.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True

        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

#重构dataset

class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None, is_transform= False):
        self.dataset = dataset
        self.transform = transform
        self.is_transform = is_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)
        if self.is_transform and self.transform is not None :
            img = self.transform(img)
        return img, pid, camid
