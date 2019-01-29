#!/usr/bin/env python3

from bs4 import BeautifulSoup as bs
from urllib.request import urlopen, urlparse, urlunparse,urlretrieve
import pandas as pd
import os
import sys

def download_picture(x, out_folder) :
    soup = bs(urlopen(x.URL))
    parsed = list(urlparse(x.URL))

    src = parsed[2].replace('/html/', 'art/').replace('.html','.jpg')
    fn = src.split('/')[-1]
    ofn = out_folder + fn

    print(src)
    urlretrieve(
        "{0}://{1}/{2}".format(parsed[0], parsed[1], src), ofn
        )
    return(src.split('/')[-1])


def print_row(x):
    print(x.URL)

out_folder = '/home/noam/bin/art_or_trash/img/unscaled/'
cat = pd.read_csv("/home/noam/bin/art_or_trash/data/catalog_mini.tab", sep='\t')
cat['name'] = cat.apply(lambda x : download_picture(x, out_folder), axis = 1)
