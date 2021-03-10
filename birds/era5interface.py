import cdsapi
import os
import os.path as osp
import argparse
import xarray as xr
import numpy as np
import pandas as pd
import rioxarray
from shapely import geometry

# from . import datahandling
# from .spatial import Spatial
from birds import datahandling
from birds.spatial import Spatial


class ERA5Loader():

    def __init__(self, radars):
        self.radars = radars

        self.surface_data_config = {'variable' : ['2m_temperature',
                                                 'surface_sensible_heat_flux',
                                                 '10m_u_component_of_wind',
                                                 '10m_v_component_of_wind',
                                                 'surface_pressure',
                                                 'total_precipitation',
                                                 ],
                                    'format' : 'netcdf',
                                    'product_type'  : 'reanalysis',}

        self.pressure_level_config = {'variable' : ['fraction_of_cloud_cover',
                                                    'specific_humidity',
                                                    #'specific_rain_water_content',
                                                    'temperature',
                                                    'u_component_of_wind',
                                                    'v_component_of_wind',
                                                    ],
                                    'pressure_level': '850',
                                    'format' : 'netcdf',
                                    'product_type' :'reanalysis',}

        self.client = cdsapi.Client()

    def download_season(self, season, year, target_dir, bounds=None, pl=850, surface_data=True):
        # load radar information
        if bounds is None:
            #radar_dir = osp.join(self.radar_path, season, year)
            #radars = datahandling.load_radars(radar_dir)
            spatial = Spatial(self.radars)
            minx, miny, maxx, maxy = spatial.cells.to_crs(epsg=spatial.epsg).total_bounds
            bounds = [maxy, minx, miny, maxx]  # North, West, South, East

        if season == 'spring':
            months = [f'{m:02}' for m in range(3, 6)]
        elif season in ['fall', 'autumn']:
            months = [f'{m:02}' for m in range(8, 12)]
        else:
            months = ['09']

        # datetime is interpreted as 00:00 UTC
        days = [f'{(d + 1):02}' for d in range(31)]
        #time = [f'{h:02}:{m:02}' for h in range(24) for m in range(0, 59, 15)]
        time = [f'{h:02}:00' for h in range(24)]
        resolution = [0.25, 0.25]

        info = { 'year' : year,
                 'month' : months,
                 'day' : days,
                 'area' : bounds,
                 'grid' : resolution, #Default: 0.25 x 0.25
                 'time' : time }

        os.makedirs(osp.dirname(target_dir), exist_ok=True)

        if surface_data:
            self.surface_data_config.update(info)
            self.client.retrieve('reanalysis-era5-land',
                                self.surface_data_config,
                                osp.join(target_dir, 'surface.nc'))
        if isinstance(pl, int):
            self.pressure_level_config.update(info)
            self.pressure_level_config['pressure_level'] = str(pl)
            self.client.retrieve('reanalysis-era5-pressure-levels',
                                 self.pressure_level_config,
                                 osp.join(target_dir, f'pressure_level_{pl}.nc'))



def extract_points(data_path, lonlat_list, t_range, vars):
    # t_range must be given as UTC
    data = xr.open_dataset(data_path)
    data = data.rio.write_crs('EPSG:4326') # set crs to lat lon
    data = data.rio.interpolate_na() # fill nan's by interpolating spatially

    #t_range = t_range.tz_convert('UTC') # convert datetimeindex to UTC if it was given at a different timezone
    weather = {}

    for var, ds in data.items():
        if var in vars:
            var_data = []
            for lonlat in lonlat_list:
                # Extract time-series data at given point (interpolate between available grid points)
                data_point = ds.interp(longitude=lonlat[0], latitude=lonlat[1], method='linear')
                var_data.append(data_point.sel(time=t_range).data.flatten())
            weather[var] = np.stack(var_data)

    return weather

def sample_point_from_polygon(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    lon = np.random.uniform(minx, maxx)
    lat = np.random.uniform(miny, maxy)
    pos = geometry.Point(lon, lat)
    while not polygon.contains(pos):
        lon = np.random.uniform(minx, maxx)
        lat = np.random.uniform(miny, maxy)
        pos = geometry.Point(lon, lat)
    return (lon, lat)

def compute_cell_avg(data_path, cell_geometries, n_points, t_range, vars):
    # t_range must be given as UTC
    data = xr.open_dataset(data_path)
    data = data.rio.write_crs('EPSG:4326') # set crs to lat lon
    data = data.rio.interpolate_na() # fill nan's by interpolating spatially

    #t_range = t_range.tz_convert('UTC') # convert datetimeindex to UTC if it was given at a different timezone
    weather = {}

    for var, ds in data.items():
        if var in vars:
            var_data = []
            for poly in cell_geometries:
                var_data_poly = []
                for i in range(n_points):
                    lon, lat = sample_point_from_polygon(poly)
                    #print(lon, lat)
                    # Extract time-series data at given point (interpolate between available grid points)
                    var_data_poly.append(ds.interp(longitude=lon,
                                                   latitude=lat,
                                                   method='linear').sel(time=t_range).data.flatten())
                var_data.append(np.nanmean(np.stack(var_data_poly, axis=0), axis=0))
            weather[var] = np.stack(var_data, axis=0)

    return weather



if __name__ == '__main__':
#
#     loader = ERA5Loader()
#     base_dir = '/home/fiona/environmental_data/era5'
#     for year in ['2015']: #'2015', '2016', '2017', '2018']:
#         for season in ['spring', 'fall']:
#             file_path = loader.download_season(year, season, osp.join(base_dir, season, year), f'wind_850.nc')



    root = '/home/fiona/birdMigration/data/raw'
    radar_path = osp.join(root, 'radar', 'fall', '2015')
    radars = datahandling.load_radars(radar_path)
    dl = ERA5Loader(radars)

    for year in ['2014', '2020']: #['2015', '2016']: #'2015', '2016', '2017', '2018']:
        for season in ['fall']: #['spring', 'fall']:
            output_dir = osp.join(root, 'env', season, year)
            os.makedirs(output_dir, exist_ok=True)
            dl.download_season(season, year, output_dir)