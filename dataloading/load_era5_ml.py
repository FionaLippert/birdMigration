import cdsapi
from cdo import Cdo
import os
import os.path as osp
import glob
import xarray as xr

import argparse


def split_level_data(input_file, output_file):
    data = xr.open_dataset(input_file)

    ds_list = []
    for var, ds in data.items():
        if 'level' in ds.coords:
            ds = ds.to_dataset(dim='level')
            ds = ds.rename({l: f'{var}_L{l}' for l in ds.data_vars})
        ds_list.append(ds)

    data = xr.merge(ds_list)
    data.to_netcdf(output_file)


home = osp.expanduser("~")

parser = argparse.ArgumentParser(description='load ERA5 model level data')
parser.add_argument('--root', type=str, default=osp.join(home, 'birdMigration', 'data'), help='entry point to required data')
parser.add_argument('--datasource', type=str, default='radar', help='datasource type (one of [abm, radar, nexrad]')
parser.add_argument('--season', type=str, default='fall', help='season to load data for (currently supports "spring" and "fall")')
parser.add_argument('--years', type=int, nargs='+', required=True)
parser.add_argument('--bounds', type=float, nargs='+', required=True)
parser.add_argument('--resolution', type=float, default=0.5) # 0.25 doesn't seem to work for specific humidity

args = parser.parse_args()

client = cdsapi.Client()

# model level variables
# temperature: 130
# u wind: 131
# v wind: 132
# specific humidity: 133
# vorticity: 138
# fraction of cloud cover: 248

if args.season == 'spring':
    months = [f'{m:02}' for m in range(3, 7)]
    levels = '135/127/115'  # corresponds to 54m, 334m, 1329m for the ICAO Standard Atmosphere
else:
    months = [f'{m:02}' for m in range(8, 12)]
    levels = '135/128/115'  # corresponds to 54m, 288m, 1329m for the ICAO Standard Atmosphere

for year in args.years:
    output_dir = osp.join(args.root, 'raw', 'env', args.datasource, args.season, str(year))
    os.makedirs(output_dir, exist_ok=True)

    for month in months:

        config = {
            'class': 'ea',
            'date': f'{year}-{month}-01/to/{year}-{month}-31',
            'expver': '1',
            'levelist': levels,
            'levtype': 'ml',
            'param': '130/131/132/133',
            'stream': 'oper',  # denotes ERA5 (vs ensemble members)
            'time': '00/to/23/by/1',
            'type': 'an',  # analysis
            'area': '/'.join([str(b) for b in args.bounds]), #'58.67/-137.21/14.34/-54.77'
            'grid': '/'.join([str(args.resolution)] * 2),
            'format': 'netcdf',
        }

        print(config)

        print(f'download data for {config["date"]}')

        client.retrieve('reanalysis-era5-complete', config,
                        osp.join(output_dir, f'{year}_{month}_model_levels.nc'))

    # use CDO to merge all files of the same year
    filenames = glob.glob(osp.join(output_dir, f'{year}_*_model_levels.nc'))
    output_file = osp.join(output_dir, f'model_levels.nc')

    cdo = Cdo()
    cdo.mergetime(input=' '.join(filenames), output=output_file, options='-b F64')

    split_level_data(output_file, output_file)