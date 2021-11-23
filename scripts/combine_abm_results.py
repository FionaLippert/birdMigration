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

# Collect results and combine them into numpy arrays
files = glob.glob(osp.join(args.path, '*.pkl'))
traj, states = [], []
print(files)
for file in files:
    with open(file, 'rb') as f:
        result = pickle.load(f)
    traj.append(result['trajectories'])
    states.append(result['states'])

traj = np.concatenate(traj, axis=1)
states = np.concatenate(states, axis=1)
time = result['time']

# remove trajectories after birds arrived in target area
print('remove traj after arrival')
target_area = gpd.read_file(args.target_area_path)
traj, states = abm.stop_birds_after_arrival(traj, states, target_area)

# write to disk
np.save(osp.join(args.path, 'traj.npy'), traj)
np.save(osp.join(args.path, 'states.npy'), states)
with open(osp.join(args.path, 'time.pkl'), 'wb') as f:
    pickle.dump(time, f)