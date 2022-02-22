import xarray as xr
import os
import os.path as osp
import argparse
import yaml
import geopandas as gpd

from birds import abm

parser = argparse.ArgumentParser(description='parallel ABM simulation')
parser.add_argument('radar_path', type=str, help='directory containing radar data as .nc files')
parser.add_argument('env_path', type=str, help='directory containing weather data as .nc file')
parser.add_argument('output_path', type=str, help='output directory')
parser.add_argument('num_birds', type=int, help='number of birds to simulate')
parser.add_argument('pid', type=int, help='process id and random seed')
parser.add_argument('departure_area_path', type=str, help='shape file of departure area')
parser.add_argument('target_area_path', type=str, help='shape file of target area')
args = parser.parse_args()

with open('abm_config.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

print(f'process {args.pid}: setup simulation')

# setup environment
start = f'{config["year"]}-{config["start_date"]}'
end = f'{config["year"]}-{config["end_date"]}'

wind_path = osp.join(args.env_path, 'pressure_level_850.nc')
wind = xr.open_dataset(wind_path).sel(time=slice(start, end))[['u', 'v']]
env = abm.Environment(wind)
print('total area', wind.longitude.max(), wind.latitude.max(), wind.longitude.min(), wind.latitude.min())

# simulation settings
settings = config['settings']
settings['num_birds'] = args.num_birds
settings['random_seed'] = args.pid

# run simulation
if len(args.departure_area_path) > 0 and len(args.target_area_path) > 0:
    departure_area = gpd.read_file(args.departure_area_path)
    target_area = gpd.read_file(args.target_area_path)
    sim = abm.Simulation(env, settings, departure_area=departure_area, target_area=target_area)
else:
    sim = abm.Simulation(env, settings)
steps = len(env.time)
print(f'process {args.pid}: start simulating for {steps} timesteps')
sim.run(steps)

# save data
print(f'process {args.pid}: save simulation results to disk')
sim.save_data(os.path.join(args.output_path, f'simulation_results_{args.pid}.pkl'))
