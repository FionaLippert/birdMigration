from birds import era5interface, datahandling
from birds.spatial import Spatial
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
parser.add_argument('--resolution', type=float, default=0.5)

args = parser.parse_args()


data_dir = osp.join(args.root, 'raw')

if args.bounds is None:
    # load radars
    if args.datasource == 'abm':
        df = gpd.read_file(osp.join(data_dir, 'abm', 'all_radars.shp'))
        radars = dict(zip(zip(df.lon, df.lat), df.radar.values))
    else:
        radar_dir = osp.join(data_dir, args.datasource, args.season, str(args.years[0]))
        radars = datahandling.load_radars(radar_dir)

    # get bounds of Voronoi tesselation
    spatial = Spatial(radars, n_dummy_radars=args.n_dummy_radars, buffer=args.voronoi_buffer)
    cells, _ = spatial.voronoi()
    minx, miny, maxx, maxy = cells.to_crs(epsg=spatial.epsg_lonlat).total_bounds
    bounds = [maxy + args.buffer_y, minx - args.buffer_x, miny - args.buffer_y, maxx + args.buffer_x]
else:
    bounds = args.bounds


single_level_config = {'variable' : [#'2m_temperature',
                                     #'surface_sensible_heat_flux',
                                     '10m_u_component_of_wind',
                                     '10m_v_component_of_wind',
                                     #'100m_u_component_of_wind',
                                     #'100m_v_component_of_wind',
                                     'surface_pressure',
                                     'mean_sea_level_pressure',
                                     'total_precipitation',
                                     #'convective_available_potential_energy',
                                     'total_cloud_cover',
                                     #'low_cloud_cover',
                                     #'medium_cloud_cover',
                                     #'high_cloud_cover',
                                     'geopotential'
                                     ],
                                    'format' : 'netcdf',
                                    'grid': [args.resolution, args.resolution],
                                    'area': bounds,
                                    'product_type': 'reanalysis',}

# pressure_level_config = {'variable' : [#'fraction_of_cloud_cover',
#                                        #'specific_humidity',
#                                        'relative_humidity',
#                                        'vorticity',
#                                        #'geopotential',
#                                        'temperature',
#                                        'u_component_of_wind',
#                                        'v_component_of_wind',
#                                        ],
#                                     'pressure_level': ['300', '400', '500', '600', '700', '800', '900', '1000'],
#                                     'format' : 'netcdf',
#                                     'resolution': [args.resolution, args.resolution],
#                                     'area': bounds,
#                                     'product_type' :'reanalysis',}

# model_level_config = {
#             'class': 'ea',
#             'expver': '1',
#             'levtype': 'ml',
#             'param': '130/131/132/133',
#             'stream': 'oper',  # denotes ERA5 (vs ensemble members)
#             'time': '00/to/23/by/1',
#             'type': 'an',  # analysis
#             'area': '/'.join([str(b) for b in args.bounds]), #'58.67/-137.21/14.34/-54.77'
#             'grid': '/'.join([str(args.resolution)] * 2),
#             'format': 'netcdf',
#         }

pressure_level_config = None
model_level_config = None

dl = era5interface.ERA5Loader(single_level_config=single_level_config,
                              pressure_level_config=pressure_level_config,
                              model_level_config=model_level_config)

for year in args.years:
    output_dir = osp.join(data_dir, 'env', args.datasource, args.season, str(year))
    os.makedirs(output_dir, exist_ok=True)
    dl.download_season(args.season, year, output_dir, bounds=args.bounds)
