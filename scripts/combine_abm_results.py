import argparse
import pickle5 as pickle
import glob
import numpy as np
import os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, help='entry point to required data')
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

# write to disk
np.save(osp.join(args.path, 'traj.npy'), traj)
np.save(osp.join(args.path, 'states.npy'), states)
with open(osp.join(args.path, 'time.pkl'), 'wb') as f:
    pickle.dump(time, f)