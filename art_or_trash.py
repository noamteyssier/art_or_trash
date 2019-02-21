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
from torch.utils.data.sampler import SubsetRandomSampler

from PIL import Image


# set dir
os.chdir("/home/noam/bin/art_or_trash/")


class ArtDataset(Dataset):
    def __init__(self, csv_filepath, img_dir, transforms):
        """create dataframe, image and label arrays, and calculate data length"""
        self.transforms = transforms
        self.df = self.__process_csv__(csv_filepath)
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

    def __process_csv__(self, csv_filepath):
        """create dataframe and embed label"""
        # read in csv
        self.df = pd.read_csv(csv_filepath, sep = "\t")

        # create parse filenames
        self.df['img_fn'] = self.df.apply(
            lambda x : x['HTML'].split('/')[-1].replace('.html','.jpg'),
            axis = 1)

        # embed categories to numeric
        self.embed = {j:i for (i,j) in enumerate(set(self.df['TYPE']))}
        self.debed = {j:i for (i,j) in self.embed.items()}

        # add as column to dataframe
        self.df['label'] = self.df.apply(lambda x : self.embed[x['TYPE']], axis = 1)

        return self.df

    def split_dataset(self, pc_train, shuffle=True):
        """shuffle indices and return 2 np arrays of indices for each set"""
        assert pc_train < 1 and pc_train > 0
        indices = np.array(range(self.__len__()))
        train_size = int(self.__len__() * pc_train)
        if shuffle == True:
            np.random.shuffle(indices)

        return indices[:train_size], indices[train_size:]
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # kernel
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.conv2 = nn.Conv2d(20, 16, 5)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 29 * 29, 240)
        self.fc2 = nn.Linear(240, 120)
        self.fc3 = nn.Linear(120, 10)

        self.pool = nn.MaxPool2d(2,2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.size())
        x = x.view(-1, 16 * 29 * 29)
        # sys.exit()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
def train(net, device, art_dataset_loader, optimizer, criterion):
    for epoch in range(10) :     # loop over the dataset multiple times

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
            if i % 200 == 199:    # print every 2000 mini-batches
                print(
                    '[%d, %d] loss: %.3f' %
                    (epoch + 1, i+1, running_loss / 200)
                    )
                running_loss = 0.0

        print('Finished Epoch : %i' %epoch)

def main():
    csv = "data/subset_catalog.tab"
    img_dir = "img/set/"

    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    art_dataset = ArtDataset(csv, img_dir, transform)
    train_indices, test_indices = art_dataset.split_dataset(0.3)
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    art_dataset_loader = torch.utils.data.DataLoader(
        dataset=art_dataset,
        batch_size = 4, num_workers = 2,
        sampler=train_sampler
        )
    test_dataset_loader = torch.utils.data.DataLoader(
        dataset=art_dataset,
        batch_size = 4, num_workers = 2,
        sampler=test_sampler
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

    train(net, device, art_dataset_loader, optimizer, criterion)
    torch.save(net.state_dict(), "/home/noam/bin/art_or_trash/aot.mdl")

    ####################
    # Test the Network #
    ####################

    dataiter = iter(test_dataset_loader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % art_dataset.debed[int(labels[j])] for j in range(4)))
    outputs = net(images.to(device))
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % art_dataset.debed[int(predicted[j])] for j in range(4)))

    # let's see how the network performs on the whole dataset
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dataset_loader:
            images, labels = [i.to(device) for i in data]
            outputs = net(images)
            _,predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images %d %%' %(
        100 * correct / total))

    # what are the classes that performed well, and the classes that didnt?
    class_correct = list(0. for _ in range(5))
    class_total = list(0. for _ in range(5))
    with torch.no_grad():
        for data in test_dataset_loader:
            images, labels = [i.to(device) for i in data]
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(5):
        print('Accuracy of %5s : %2d %%' % (
            art_dataset.debed[i], 100 * class_correct[i] / class_total[i]))

if __name__ == '__main__':
    main()
