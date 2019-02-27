#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns
import os

os.chdir("/home/noam/bin/art_or_trash/")

def main():
    models = pd.concat(
        [pd.read_csv("models/" + i, sep="\t") for i in os.listdir("models/") if ".log" in i])

    models.learning_rate = models.learning_rate.astype('float')
    models.momentum = models.momentum.astype('float')
    models.overall_accuracy = models.overall_accuracy.astype('float')


    sns.relplot(data=models, x='epoch', y='overall_accuracy', col='learning_rate', kind='line', hue='momentum')
    sns.relplot(data=models, x='epoch', y='overall_accuracy', col='momentum', kind='line', hue='learning_rate')
    sns.relplot(data=models, x='epoch', y='overall_accuracy', row='learning_rate', col='momentum', kind='line')


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

if __name__ == '__main__':
    main()
