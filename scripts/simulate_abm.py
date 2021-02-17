import xarray as xr
import os
import os.path as osp
import sys
import argparse
import yaml
import geopandas as gpd

sys.path.insert(1, osp.join(sys.path[0], '../modules'))
from birds import abm
from birds import datahandling
from birds.spatial import Spatial

parser = argparse.ArgumentParser(description='parallel ABM simulation')
parser.add_argument('radar_path', type=str, help='directory containing radar data as .nc files')
parser.add_argument('wind_path', type=str, help='directory containing wind data as .nc file')
parser.add_argument('output_path', type=str, help='output directory')
parser.add_argument('num_birds', type=int, help='number of birds to simulate')
parser.add_argument('pid', type=int, help='process id and random seed')
parser.add_argument('departure_area_path', type=str, help='output directory')
args = parser.parse_args()

with open('abm_config.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

print(f'process {args.pid}: setup simulation')

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
settings['random_seed'] = args.pid

# run simulation
if len(args.departure_area_path) > 0:
    area = gpd.read_file(args.departure_area_path)
    sim = abm.Simulation(env, buffers, settings, departure_area=area)
else:
    sim = abm.Simulation(env, buffers, settings)
steps = len(env.time)
print(f'process {args.pid}: start simulating for {steps} timesteps')
sim.run(steps)

# save data
print(f'process {args.pid}: save simulation results to disk')
sim.save_data(os.path.join(args.output_path, f'simulation_results_{args.pid}.pkl'))
sim.data.plot_trajectories(os.path.join(args.output_path, f'trajectories_{args.pid}.png'))
