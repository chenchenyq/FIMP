import cv2
import numpy as np
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''

    def __init__(self, dataset, file_root='data/', transform=None):
        """
        dataset: dataset name, e.g. NJU2K_NLPR_train
        file_root: root of data_path, e.g. ./data/
        """
        self.file_list = open(file_root + '/list/' + dataset + '.txt').read().splitlines()
        self.pre_images = [file_root + '/A/' + x + '.png' for x in self.file_list]
        self.post_images = [file_root + '/B/' + x + '.png' for x in self.file_list]
        self.gts = [file_root + '/label/' + x + '.png' for x in self.file_list]
        self.transform = transform

    def __len__(self):
        return len(self.pre_images)


    def __getitem__(self, idx):
        name = self.file_list[idx]

        pre_image_name = self.pre_images[idx]
        label_name = self.gts[idx]
        post_image_name = self.post_images[idx]

        pre_image = cv2.imread(pre_image_name)
        post_image = cv2.imread(post_image_name)
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)

        img = np.concatenate((pre_image, post_image), axis=2)

        if self.transform:
            [img, label] = self.transform(img, label)

        return img, label, name


