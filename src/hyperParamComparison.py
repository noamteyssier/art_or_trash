#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
import sys

os.chdir("/home/noam/bin/art_or_trash/")



def plot_lr(models):
    """plot learning rate with hued momentum"""
    sns.relplot(data=models, x='epoch', y='overall_accuracy', col='learning_rate', kind='line', hue='momentum')
    plt.show()
def plot_m(models):
    """plot momentum with hued learning rate"""
    sns.relplot(data=models, x='epoch', y='overall_accuracy', col='momentum', kind='line', hue='learning_rate')
    plt.show()
def plot_lrm(models):
    """plot separated learning rate and momentum"""
    sns.relplot(data=models, x='epoch', y='overall_accuracy', row='learning_rate', col='momentum', kind='line')
    plt.show()
def plot_class_accuracy(models):
    """plot class accuracies and overall accuracy"""
    class_diff = pd.melt(
        models,
        id_vars=['epoch', 'learning_rate', 'momentum'],
        value_vars=models.columns[3:],
        var_name="image_class",
        value_name="accuracy"
        )

    sns.relplot(
        data = class_diff,
        x = 'epoch',
        y = 'accuracy',
        col = 'learning_rate',
        row = 'momentum',
        kind='line',
        hue='image_class')
    plt.show()
def max_accuracies_by_pair(models):
    """show best pair of learning rate, momentum, epoch for highest mean class accuracy"""
    class_diff = pd.melt(
        models,
        id_vars=['epoch', 'learning_rate', 'momentum'],
        value_vars=models.columns[3:],
        var_name="image_class",
        value_name="accuracy"
        )
    mean_acc = class_diff[class_diff.image_class != 'overall_accuracy'].\
        groupby(['learning_rate', 'momentum','epoch']).\
        mean().\
        reset_index().\
        sort_values('accuracy', ascending=False).\
        drop_duplicates(['learning_rate', 'momentum'])
    mean_acc.to_csv(sys.stdout, sep="\t", index=False)
def main():
    # load model logs
    models = pd.concat([pd.read_csv("models/" + i, sep="\t") for i in os.listdir("models/") if ".log" in i])

    plot_lr(models)
    plot_m(models)
    plot_lrm(models)
    plot_class_accuracy(models)
    max_accuracies_by_pair(models)

if __name__ == '__main__':
    main()
