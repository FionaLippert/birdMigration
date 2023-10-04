from birds import era5interface, datahandling
import geopandas as gpd
import os.path as osp
import os
import argparse


home = osp.expanduser("~")

parser = argparse.ArgumentParser(description='load ERA5 data')
parser.add_argument('--root', type=str, default=osp.join(home, 'birdMigration', 'data'), help='entry point to required data')
parser.add_argument('--datasource', type=str, default='radar', help='datasource type (one of [abm, radar, nexrad]')
parser.add_argument('--buffer_x', type=int, default=4, help='longitude buffer around voronoi area')
parser.add_argument('--buffer_y', type=int, default=4, help='latitude buffer around voronoi area')
parser.add_argument('--season', type=str, default='fall', help='season to load data for (currently supports "spring" and "fall")')
parser.add_argument('--years', type=int, nargs='+', required=True)
parser.add_argument('--n_dummy_radars', type=int, default=0)
parser.add_argument('--voronoi_buffer', type=int, default=150_000)
parser.add_argument('--bounds', type=float, nargs='+', default=None)

args = parser.parse_args()


data_dir = osp.join(args.root, 'raw')

# load radars
if args.datasource == 'abm':
    df = gpd.read_file(osp.join(data_dir, 'abm', 'all_radars.shp'))
    radars = dict(zip(zip(df.lon, df.lat), df.radar.values))
else:
    radar_dir = osp.join(data_dir, args.datasource, args.season, str(args.years[0]))
    radars = datahandling.load_radars(radar_dir)

dl = era5interface.ERA5Loader(radars)

# minx, miny, maxx, maxy = df.total_bounds
# bounds = [maxy + args.buffer_y, # North
#           minx - args.buffer_x, # West
#           miny - args.buffer_y, # South
#           maxx + args.buffer_x] # East

for year in args.years:
    output_dir = osp.join(data_dir, 'env', args.datasource, args.season, str(year))
    os.makedirs(output_dir, exist_ok=True)
    dl.download_season(args.season, year, output_dir, pl=850, surface_data=True,
                       buffer_x=args.buffer_x, buffer_y=args.buffer_y,
                       n_dummy_radars=args.n_dummy_radars, buffer=args.voronoi_buffer,
                       bounds=args.bounds)
