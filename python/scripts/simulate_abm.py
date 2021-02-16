import xarray as xr
import os
import os.path as osp
import sys
import argparse
import yaml

sys.path.insert(1, osp.join(sys.path[0], '../modules'))
import abm
import datahandling
from spatial import Spatial

parser = argparse.ArgumentParser(description='parallel ABM simulation')
parser.add_argument('radar_path', type=str, help='directory containing radar data as .nc files')
parser.add_argument('wind_path', type=str, help='directory containing wind data as .nc file')
parser.add_argument('output_path', type=str, help='output directory')
parser.add_argument('num_birds', type=int, help='number of birds to simulate')
parser.add_argument('seed', type=int, help='random seed')
args = parser.parse_args()

with open('abm_config.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# get 25km buffers around radar stations to simulate VP measurements
radars = datahandling.load_radars(args.radar_path)
sp = Spatial(radars)
buffers = sp.pts_local.buffer(25_000).to_crs(epsg=sp.epsg).to_dict()

# setup environment
start = f'{config["year"]}-{config["start_date"]}'
end = f'{config["year"]}-{config["end_date"]}'
wind = xr.open_dataset(args.wind_path).sel(time=slice(start, end))
env = abm.Environment(wind)

# simulation settings
settings = config['settings']
settings['num_birds'] = args.num_birds
settings['random_seed'] = args.seed

# run simulation
sim = abm.Simulation(env, buffers, settings)
steps = len(env.time)
sim.run(steps)

# save data
sim.save_data(os.path.join(args.output_path, f'simulation_results_{args.seed}.pkl'))
sim.data.plot_trajectories(os.path.join(args.output_path, f'trajectories_{args.seed}.png'))
