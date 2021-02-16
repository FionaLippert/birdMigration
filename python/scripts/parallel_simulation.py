import subprocess
import yaml
import argparse
from datetime import datetime
import numpy as np
import multiprocessing as mp
import os
import os.path as osp
import sys

sys.path.insert(1, osp.join(sys.path[0], '../modules'))
import datahandling
from spatial import Spatial
from era5interface import ERA5Loader


###################################### SETUP ##################################################
# load config file
config_file = 'abm_config.yml'
with open(config_file) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# file paths
root = '/home/fiona/birdMigration/data'
wind_path = osp.join(root, 'raw', 'wind', config['season'], config['year'], 'wind_850.nc')
radar_path = osp.join(root, 'raw', 'radar', config['season'], config['year'])
output_path = osp.join(root, 'experiments', 'abm', config['season'], config['year'], f'experiment_{datetime.now()}')
os.makedirs(output_path, exist_ok=True)

with open(os.path.join(output_path, config_file), 'w+') as f:
    yaml.dump(config, f)

# if wind data is not available, download it
if not osp.exists(wind_path):
    radars = datahandling.load_radars(radar_path)
    sp = Spatial(radars)
    minx, miny, maxx, maxy = sp.cells.to_crs(epsg=sp.epsg).total_bounds
    bounds = [maxy, minx, miny, maxx] # North, West, South, East
    ERA5Loader().download_season(config['year'], config['season'], wind_path, bounds)

##################################### END SETUP ################################################


start_time = datetime.now()
processes = set()
num_processes = mp.cpu_count() - 2

# log file
logfile = os.path.join(output_path, 'log.txt')

N = config['num_birds']
birds_pp = [int(N/num_processes) for p in range(num_processes)]
for r in range(N % num_processes):
    birds_pp[r] += 1

for p in range(num_processes):
    print(f'---------- start simulating {birds_pp[p]} birds ------------')
    processes.add(subprocess.Popen(['python', 'simulate_abm.py',
                                    radar_path, wind_path, output_path, str(birds_pp[p]), str(p)],
                                   stdout=open(logfile, 'a+'),
                                   stderr=open(logfile, 'a+')))

# Check if all the child processes were closed
for p in processes:
    if p.poll() is None:
        p.wait()

time_elapsed = datetime.now() - start_time
with open(logfile, 'a+') as f:
    f.write('\n')
    f.write(f'Time elapsed (hh:mm:ss.ms) {time_elapsed} \n')