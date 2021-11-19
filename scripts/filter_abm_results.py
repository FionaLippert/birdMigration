import argparse
import pickle5 as pickle
import glob
import numpy as np
import os.path as osp
import geopandas as gpd
from birds import abm

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, help='entry point to required data')
parser.add_argument('target_area', type=str, help='shape file of target area')
args = parser.parse_args()

# load abm results
traj = np.load(osp.join(args.path, 'traj.npy'))
states = np.load(osp.join(args.path, 'states.npy'))

# remove trajectories after birds arrived in target area
print('remove traj after arrival')
target_area = gpd.read_file(args.target_area_path)
traj, states = abm.stop_birds_after_arrival(traj, states, target_area)

# write to disk
np.save(osp.join(args.path, 'traj_filtered.npy'), traj)
np.save(osp.join(args.path, 'states_filtered.npy'), states)