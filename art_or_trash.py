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
    def __init__(self, csv_filepath, img_dir, transforms):
        """create dataframe, image and label arrays, and calculate data length"""
        self.transforms = transforms
        self.df = self.process_csv(csv_filepath)
        self.img_dir = img_dir
        self.image_array = np.asarray(self.df['img_fn'])
        self.label_array = np.asarray(self.df['label'])
        self.data_len = len(self.df.index)
    def __getitem__(self, index):
        """apply transform and return image and corresponding label"""
        single_image_name = self.img_dir + self.image_array[index]
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

        # create parse filenames
        self.df['img_fn'] = self.df.apply(
            lambda x : x['HTML'].split('/')[-1].replace('.html','.jpg'),
            axis = 1)

        # embed categories to numeric
        self.embed = {j:i for (i,j) in enumerate(set(self.df['TYPE']))}

        # add as column to dataframe
        self.df['label'] = self.df.apply(lambda x : self.embed[x['TYPE']], axis = 1)

        return self.df
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # kernel
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.conv2 = nn.Conv2d(20, 16, 5)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(829440, 256)
        self.fc2 = nn.Linear(256, 100)
        self.fc3 = nn.Linear(100, 10)

        self.pool = nn.MaxPool2d(2,2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 810 * 32 * 32)
        # print(x.size())
        # sys.exit()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


csv = "data/current_catalog.tab"
img_dir = "img/train/"

transform = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
art_dataset = ArtDataset(csv, img_dir, transform)
art_dataset_loader = torch.utils.data.DataLoader(
    dataset=art_dataset,
    batch_size = 10, num_workers = 2,
    shuffle=True
    )


net = Net()
# transfer network to gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)


########################################
# Define a Loss Function and Optimizer #
########################################

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

#####################
# Train the Network #
#####################

for epoch in range(2) :     # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(art_dataset_loader, 0):
        inputs, labels = [il for il in data]

        # get the inputs
        inputs, labels = [il.to(device) for il in data]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(
                '[%d, %d] loss: %.3f' %
                (epoch + 1, i+1, running_loss / 2000)
                )
            running_loss = 0.0

print('Finished Training')
