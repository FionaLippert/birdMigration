import subprocess
import yaml
import argparse
from datetime import datetime
import numpy as np
import multiprocessing as mp
import os
import os.path as osp
import sys
from shapely.geometry import Polygon, Point
from shapely.ops import cascaded_union
import geopandas as gpd
import glob
import pickle5 as pickle

#sys.path.insert(1, osp.join(sys.path[0], '../modules'))
from birds import datahandling
from birds.spatial import Spatial
from birds.era5interface import ERA5Loader

import argparse

parser = argparse.ArgumentParser(description='parallel ABM simulation')
parser.add_argument('--root', type=str, default='/home/fiona/birdMigration/data', help='entry point to required data')
parser.add_argument('--buffer_x', type=int, default=0, help='longitude buffer around voronoi area')
parser.add_argument('--buffer_y', type=int, default=0, help='latitude buffer around voronoi area')
args = parser.parse_args()


###################################### SETUP ##################################################
# load config file
config_file = 'abm_config.yml'
with open(config_file) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# file paths
#root = '/home/fiona/birdMigration/data'
root = args.root
env_path = osp.join(root, 'raw', 'env', config['season'], config['year'],
                    f'buffer_{args.buffer_x}_{args.buffer_y}') #, 'wind_850.nc')
radar_path = osp.join(root, 'raw', 'radar', config['season'], '2015') #config['year'])
output_path = osp.join(root, 'experiments', 'abm', config['season'], config['year'], f'experiment_{datetime.now()}')
departure_area_path = osp.join(root, 'shapes', 'departure_area.shp')
target_area_path = osp.join(root, 'shapes', 'target_area.shp')
os.makedirs(output_path, exist_ok=True)

with open(os.path.join(output_path, config_file), 'w+') as f:
    yaml.dump(config, f)

# if wind data is not available, download it
if not osp.exists(env_path):
    radars = datahandling.load_radars(radar_path)
    dl = ERA5Loader(radars)
    dl.download_season(config['season'], config['year'], env_path,
                       pl=850, buffer_x=args.buffer_x, buffer_y=args.buffer_y, surface_data=True)

if not osp.exists(departure_area_path):
    countries = gpd.read_file(osp.join(root, 'shapes', 'ne_10m_admin_0_countries_lakes.shp'))
    roi = countries[countries['ADMIN'].isin(['Germany', 'Belgium', 'Netherlands'])]
    outer = cascaded_union(roi.geometry)
    inner = gpd.GeoSeries(outer, crs='epsg:4326').to_crs('epsg:3035').buffer(-50_000).to_crs('epsg:4326')
    diff = outer.difference(inner.geometry[0])

    minx = 4
    maxx = 15.1
    miny_west = 53
    miny_east = 50.5
    maxy = 55.1

    poly = Polygon([(minx, maxy), (maxx, maxy), (maxx, miny_east), (minx, miny_west)])
    area = gpd.GeoSeries(diff.intersection(poly))
    area.to_file(departure_area_path)

if not osp.exists(target_area_path):
    print('load target area')
    countries = gpd.read_file(osp.join(root, 'shapes', 'ne_10m_admin_0_countries_lakes.shp'))
    print(countries['ADMIN'])
    roi = countries[countries['ADMIN'].isin(['France', 'Spain', 'Andorra'])]
    print(roi['ADMIN'])
    outer = cascaded_union(roi.geometry)
    print(outer)

    minx = -2.5
    maxx = 4
    miny_east = 41.8
    miny_west = 43.3
    maxy_east = 42.5
    maxy_west = 44

    poly = Polygon([(minx, maxy_west), (maxx, maxy_east), (maxx, miny_east), (minx, miny_west)])
    print(poly)
    target_area = gpd.GeoSeries(outer.intersection(poly))
    print('save target area')
    area.to_file(target_area_path)


##################################### END SETUP ################################################


start_time = datetime.now()
processes = set()
num_processes = mp.cpu_count() - 3

# log file
logfile = os.path.join(output_path, 'log.txt')

N = config['num_birds']
birds_pp = [int(N/num_processes) for p in range(num_processes)]
for r in range(N % num_processes):
    birds_pp[r] += 1

for p in range(num_processes):
    print(f'---------- start simulating {birds_pp[p]} birds ------------')
    processes.add(subprocess.Popen(['python', 'simulate_abm.py',
                                    radar_path, env_path, output_path, str(birds_pp[p]), str(p),
                                    departure_area_path, target_area_path],
                                   stdout=open(logfile, 'a+'),
                                   stderr=open(logfile, 'a+')))

# Check if all the child processes were closed
for p in processes:
    if p.poll() is None:
        p.wait()

# Collect results and combine them into numpy arrays
files = glob.glob(osp.join(output_path, '*.pkl'))
traj, states, directions, speeds = [], [], [], []
print(files)
for file in files:
    with open(file, 'rb') as f:
        result = pickle.load(f)
    traj.append(result['trajectories'])
    states.append(result['states'])
    directions.append(result['directions'])
    speeds.append(result['ground_speeds'])

traj = np.concatenate(traj, axis=1)
states = np.concatenate(states, axis=1)
directions = np.concatenate(directions, axis=1)
speeds = np.concatenate(speeds, axis=1)
time = result['time']
print(time)

# write to disk
np.save(osp.join(output_path, 'traj.npy'), traj)
np.save(osp.join(output_path, 'states.npy'), states)
np.save(osp.join(output_path, 'directions.npy'), directions)
np.save(osp.join(output_path, 'ground_speeds.npy'), speeds)
with open(osp.join(output_path, 'time.pkl'), 'wb') as f:
    pickle.dump(time, f)


time_elapsed = datetime.now() - start_time
with open(logfile, 'a+') as f:
    f.write('\n')
    f.write(f'Time elapsed (hh:mm:ss.ms) {time_elapsed} \n')
