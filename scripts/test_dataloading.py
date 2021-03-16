
from birds.graphNN import *
import argparse
from datetime import datetime
from matplotlib import pyplot as plt
import itertools as it
import os
import os.path as osp
import pickle5 as pickle
import pandas as pd
import torch
from torch_geometric.data import DataLoader
from torch.optim import lr_scheduler
from birds import GBT, datasets
import networkx as nx
import geopandas as gpd

data_source = 'abm'
root = '/home/fiona/birdMigration'
data_root = osp.join(root, 'data')
gam_csv = osp.join(data_root, 'seasonal_trends', f'gam_summary_{data_source}.csv')

season = 'fall'
train_years = ['2016', '2017', '2018']
val_year = '2019'
test_year = '2015'

bird_scale = 2000
timesteps = 6


# old loader
train_data_old = RadarData(data_root, 'train', train_years[0], season, timesteps, data_source=data_source,
                            use_buffers=False, bird_scale=bird_scale)

preprocessed_dir = osp.join(data_root, 'preprocessed', data_source, season, train_years[0])
# new data
train_data_new = datasets.RadarData(data_root, 'train', train_years[0], season, timesteps, data_source=data_source,
                            use_buffers=False, bird_scale=bird_scale, old=False)

print(train_data_old[5].coords)
print(train_data_new[5].coords)