#!/usr/bin/env python3

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image

import argparse
import sys
import os

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
        self.embed = {j:i for (i,j) in enumerate(self.df['TYPE'].unique())}
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
        x = x.view(-1, 16 * 29 * 29)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class AOT():
    def __init__(self, args):
        self.catalog = "data/mini_subset_catalog.tab"
        self.img_dir = "img/set/"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.args = args

        self.net = None
        self.transform = None
        self.art_dataset = None
        self.num_classes = None
        self.class_labels = None
        self.train_loader = None
        self.test_loader = None
        self.criterion = None
        self.optimizer = None
        self.test_results = list()

        self.__init_args__()
        self.__init_network__()
        self.__init_transform__()
        self.__init_art_dataset_loaders__()
        self.__init_criterion__()
        self.__init_optimizer__()
    def __init_network__(self):
        """initialize network and move to device"""
        self.net = Net().to(self.device)
    def __init_transform__(self):
        """initialize image transformation"""
        self.transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    def __init_art_dataset_loaders__(self):
        """initalize art dataset and loaders"""
        self.art_dataset = ArtDataset(self.catalog, self.img_dir, self.transform)
        self.num_classes = len(self.art_dataset.debed.items())
        self.class_labels = [self.art_dataset.debed[i] for i in range(self.num_classes)]

        # split dataset randomly and create loaders
        train_indices, test_indices = self.art_dataset.split_dataset(self.args.train_split)
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.art_dataset,
            batch_size = 4, num_workers = 7,
            sampler=train_sampler)
        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.art_dataset,
            batch_size = 4, num_workers = 7,
            sampler=test_sampler)
    def __init_criterion__(self):
        """initialize criterion for model training"""
        self.criterion = nn.CrossEntropyLoss()
    def __init_optimizer__(self):
        """initialize optimizer for model training"""
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr = self.args.learning_rate,
            momentum = self.args.momentum)
    def __init_args__(self):
        """modify arguments to reflect model defaults"""
        if self.args.path == 'INTERNAL' :
            self.args.path = "models/mdl_{0}lr_{1}m.pt".format(self.args.learning_rate, self.args.momentum)
    def save_model_params(self):
        """save model parameters to path specified"""
        torch.save(self.net.state_dict(), self.args.path)
        self.write_log()
    def load_model_params(self):
        """load model parameters from save"""
        self.net.load_state_dict(
            torch.load(self.args.path))
        self.net.eval()
        self.net.to(self.device)
    def train(self):
        """train model on dataset and generate log of tests"""
        run_epochs = list()
        for epoch in range(1, self.args.epochs + 1) :     # loop over the dataset multiple times
            self.current_epoch = epoch
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                inputs, labels = [il for il in data]

                # get the inputs
                inputs, labels = [il.to(self.device) for il in data]

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 200 == 199:    # print every 2000 mini-batches
                    sys.stderr.write(
                        '[%d, %d] loss: %.3f\n' %
                        (epoch, i+1, running_loss / 200)
                        )
                    running_loss = 0.0
            sys.stderr.write('Finished Epoch : %i\n' %epoch)

            if self.current_epoch % self.args.test_rate == 0:
                self.test()
                run_epochs.append(self.current_epoch)
        if self.current_epoch not in run_epochs:
            self.test()
        self.save_model_params()
    def test(self):
        """test model on dataset and store accuracies"""
        class_correct = list(0. for _ in range(self.num_classes))
        class_total = list(0. for _ in range(self.num_classes))
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = [i.to(self.device) for i in data]
                outputs = self.net(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        acc = correct / total
        class_acc = [class_correct[i] / class_total[i] for i in range(self.num_classes)]

        self.test_results.append(
            [self.current_epoch, self.args.learning_rate, self.args.momentum] + [acc] + class_acc)
    def write_log(self):
        """write hyperparams and accuracies to log"""
        results = pd.DataFrame(self.test_results)
        results.columns = ['epoch', 'learning_rate', 'momentum', 'overall_accuracy'] + self.class_labels
        results.to_csv(self.args.path.replace('.pt', '.log'), sep="\t", index=False)
    def test_image(self):
        """predict given image label"""
        image = self.transform(Image.open(self.args.image).convert('RGB'))
        images = [image.to(self.device) for _ in range(4)]
        images = torch.stack(images)
        with torch.no_grad():
            output = self.net(images)
            _, predicted = torch.max(output, 1)
            label = self.class_labels[int(predicted[0])]
            print("Predicted : {0}".format(label))

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", '--image',
        help="input image to test")
    p.add_argument('-t', '--train', action='store_true',
        help="training flag to retrain model")
    p.add_argument('-l', '--learning_rate', default = 0.001, type=float,
        help='learning rate of the model training')
    p.add_argument('-m', '--momentum', default=0.9, type=float,
        help = 'momentum of learning')
    p.add_argument('-e', '--epochs', default=20, type=int,
        help = 'number of epochs to iterate')
    p.add_argument('-s', '--train_split', default=0.3, type=float,
        help='percentage of the set to use for training')
    p.add_argument('-r', '--test_rate', default=5, type=int,
        help="number of epochs to test model with")
    p.add_argument('-p', '--path', default="INTERNAL",
        help="path to save model to or load from")
    p.add_argument('-z', '--optimize', action='store_true',
        help='run model to optimize hyperparameters')
    args = p.parse_args()

    # if no args given print help
    if len(sys.argv) == 1:
        p.print_help(sys.stderr)
        sys.exit(1)

    return args
def main():
    args = get_args()

    if args.optimize:
        l_list = [0.0001, 0.0005, 0.001, 0.005, 0.01]
        m_list = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for l in l_list:
            for m in m_list:
                args.learning_rate = l
                args.momentum = m
                args.path = 'INTERNAL'
                a = AOT(args)
                a.train()

    aot = AOT(args)
    if args.train:
        aot.train()
    if args.image:
        aot.load_model_params()
        aot.test_image()



if __name__ == '__main__':
    main()
