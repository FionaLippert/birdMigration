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

#sys.path.insert(1, osp.join(sys.path[0], '../modules'))
from birds import datahandling
from birds.spatial import Spatial
from birds.era5interface import ERA5Loader

import argparse

parser = argparse.ArgumentParser(description='parallel ABM simulation')
parser.add_argument('--root', type=str, default='/home/fiona/birdMigration/data', help='entry point to required data')
args = parser.parse_args()


###################################### SETUP ##################################################
# load config file
config_file = 'abm_config.yml'
with open(config_file) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# file paths
#root = '/home/fiona/birdMigration/data'
root = args.root
env_path = osp.join(root, 'raw', 'env', config['season'], config['year']) #, 'wind_850.nc')
radar_path = osp.join(root, 'raw', 'radar', config['season'], config['year'])
output_path = osp.join(root, 'experiments', 'abm', config['season'], config['year'], f'experiment_{datetime.now()}')
departure_area_path = osp.join(root, 'shapes', 'departure_area.shp')
os.makedirs(output_path, exist_ok=True)

with open(os.path.join(output_path, config_file), 'w+') as f:
    yaml.dump(config, f)

# if wind data is not available, download it
if not osp.exists(env_path):
    dl = ERA5Loader(radar_path)
    dl.download_season(config['season'], config['year'], env_path, pl=850, surface_data=True)

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
                                    departure_area_path],
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