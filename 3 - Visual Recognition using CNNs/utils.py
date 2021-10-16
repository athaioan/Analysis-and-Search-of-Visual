import pickle
import numpy as np
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


def construct_data(data_files,data_folder):

    data = []
    labels = []
    for index, current_batch in enumerate(data_files):
        current_data = unpickle(data_folder + current_batch)
        if index != 0:
            data = data + current_data[b'data'].tolist()
            labels = labels + current_data[b'labels']
        else:
            data = current_data[b'data'].tolist()
            labels = current_data[b'labels']

    return np.asarray(data), np.asarray(labels)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class Cifar_Dataset(Dataset):

    def __init__(self, imgs, labels):

        self.imgs = imgs
        self.labels = labels



    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):

        img = self.imgs[idx, :]


        img = np.reshape(img, (3, 32, 32))

        img = img/255 - 0.5 # normalizing image from 0-255 to [-0.5,0.5]

        label = self.labels[idx]

        img = torch.from_numpy(img).cuda()
        img = img.type(torch.float32)

        label = torch.from_numpy(np.asarray(label)).cuda()
        label = label.type(torch.int64)

        return img, label
