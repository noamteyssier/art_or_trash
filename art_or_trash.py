#!/usr/bin/env python3

import torch
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset

from PIL import Image


# set dir
os.chdir("/home/noam/bin/art_or_trash/")


class ArtDataset(Dataset):
    def __init__(self, csv_filepath, transforms):
        """create dataframe, image and label arrays, and calculate data length"""
        self.transforms = transforms

        self.df = self.process_csv(csv_filepath)
        self.image_array = np.asarray(self.df['full_path'])
        self.label_array = np.asarray(self.df['label'])
        self.data_len = len(self.df.index)
    def __getitem__(self, index):
        """apply transform and return image and corresponding label"""
        single_image_name = self.image_array[index]
        single_image_label = self.label_array[index]

        img_as_img = Image.open(single_image_name).convert('RGB')
        transformed_image = self.transforms(img_as_img)

        return (transformed_image, single_image_label)
    def __len__(self):
        '''return count of dataset'''
        return self.data_len


    def process_csv(self, csv_filepath):
        """create dataframe and embed label"""
        # read in csv
        self.df = pd.read_csv(csv_filepath, sep = "\t")

        # embed categories to numeric
        self.embed = {j:i for (i,j) in enumerate(set(self.df['TYPE']))}

        # add as column to dataframe
        self.df['label'] = self.df.apply(lambda x : self.embed[x['TYPE']], axis = 1)

        return self.df

csv = "data/current_catalog_min.tab"
transform = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
art_dataset = ArtDataset(csv, transform)
art_dataset_loader = torch.utils.data.DataLoader(
    dataset=art_dataset,
    batch_size = 10,
    shuffle=True
    )

for images,labels in art_dataset_loader:
    print(labels)
    break
