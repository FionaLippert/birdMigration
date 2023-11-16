import os.path as osp
import os
from osgeo import gdal, osr
import xarray as xr
import rioxarray
import numpy as np
from shapely import geometry
import geopandas as gpd
import argparse


home = osp.expanduser("~")

parser = argparse.ArgumentParser(description='load land cover data')
parser.add_argument('--root', type=str, default=osp.join(home, 'birdMigration', 'data'), help='entry point to required data')
parser.add_argument('--datasource', type=str, default='radar', help='datasource type (one of [abm, radar, nexrad]')
parser.add_argument('--season', type=str, default='fall', help='season to load data for (currently supports "spring" and "fall")')
parser.add_argument('--years', type=int, nargs='+', required=True)
parser.add_argument('--convert2lonlat', action='store_true')
parser.add_argument('--coarsen', type=int, default=1)

args = parser.parse_args()


data_dir = osp.join(args.root, 'raw')

for year in args.years:
    output_dir = osp.join(data_dir, 'landcover', args.datasource, args.season, str(year))
    local_dir = osp.join(data_dir, 'landcover', args.datasource, 'local_proj', str(year))
    lonlat_dir = osp.join(data_dir, 'landcover', args.datasource, 'lonlat_proj', str(year))

    os.makedirs(output_dir, exist_ok=True)

    if args.convert2lonlat:

        for file in os.listdir(local_dir):
            if file.endswith(('.img')):
                print(f'loading file {file} (local crs)')
                raster = gdal.Open(osp.join(local_dir, file))

                print(f'converting to lonlat crs')
                os.makedirs(lonlat_dir, exist_ok=True)
                lonlat_raster = osp.join(lonlat_dir, 'lonlat_raster.tiff')
                # driver = gdal.GetDriverByName("GTiff")
                warp = gdal.Warp(lonlat_raster, raster, dstSRS='EPSG:4326', format="GTiff")
                warp = None

                break

    for file in os.listdir(lonlat_dir):
        if file.endswith(('.tiff')):
            print(f'loading file {file} (lonlat crs)')
            data = rioxarray.open_rasterio(osp.join(lonlat_dir, file))
            print(data)
            print(data.rio.crs)
            data.rio.write_crs('EPSG:4326')

            break

    if args.coarsen > 1:

        # def get_majority_class(data):
        #     values, counts = np.unique(data, return_counts=True)
        #     idx = np.argmax(counts)
        #     return values[idx]
        # data =  data.coarsen(x=args.coarsen, y=args.coarsen, boundary='pad').reduce(get_majority_class)

        data = data.coarsen(x=args.coarsen, y=args.coarsen, boundary='pad').count()

    data.rio.to_raster(osp.join(output_dir, 'lonlat_raster.tiff'))

    # gdf = data.to_pandas().stack().reset_index()