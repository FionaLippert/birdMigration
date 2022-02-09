from birds import era5interface
import geopandas as gpd
import os.path as osp
import os
import argparse


home = osp.expanduser("~")

parser = argparse.ArgumentParser(description='load ERA5 data')
parser.add_argument('--root', type=str, default=osp.join(home, 'birdMigration', 'data'), help='entry point to required data')
parser.add_argument('--buffer_x', type=int, default=4, help='longitude buffer around voronoi area')
parser.add_argument('--buffer_y', type=int, default=4, help='latitude buffer around voronoi area')
parser.add_argument('--season', type=str, default='fall', help='season to load data for (currently supports "spring" and "fall")')
parser.add_argument('--years', type=int, nargs='+', required=True)
args = parser.parse_args()


data_dir = osp.join(args.root, 'raw')
df = gpd.read_file(osp.join(data_dir, 'abm', 'all_radars.shp'))
radars = dict(zip(zip(df.lon, df.lat), df.radar.values))
dl = era5interface.ERA5Loader(radars)

minx, miny, maxx, maxy = df.total_bounds
bounds = [maxy + args.buffer_y, # North
          minx - args.buffer_x, # West
          miny - args.buffer_y, # South
          maxx + args.buffer_x] # East

for year in args.years:
    output_dir = osp.join(data_dir, 'env', args.season, year)
    os.makedirs(output_dir, exist_ok=True)
    dl.download_season(args.season, year, output_dir, pl=850, bounds=bounds, surface_data=True)
