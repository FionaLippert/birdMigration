#!/usr/bin/env python
import cdsapi
import os
import argparse
import datahandling
import xarray as xr
import numpy as np
import rioxarray
import pandas as pd


class ERA5Loader():

    def __init__(self, radar_path='/home/fiona/radar_data/vpi/night_only'):
        self.radar_path = radar_path

        self.surface_data_config = {'variable' : [#'2m_temperature',
                                                 'surface_sensible_heat_flux',
                                                 #'10m_u_component_of_wind',
                                                 #'10m_v_component_of_wind',
                                                 'surface_pressure',
                                                 'total_precipitation',
                                                 ],
                                    'format' : 'netcdf',
                                    'product_type'  : 'reanalysis',}

        self.pressure_level_config = {'variable' : [#'fraction_of_cloud_cover',
                                                    #'specific_humidity',
                                                    #'specific_rain_water_content',
                                                    #'temperature',
                                                    'u_component_of_wind',
                                                    'v_component_of_wind',
                                                    ],
                                    'pressure_level': '850',
                                    'format' : 'netcdf',
                                    'product_type' :'reanalysis',}

        self.client = cdsapi.Client()

    def download_season(self, year, season, target_dir, filename):
        # load radar information
        if season == 'spring':
            months = [f'{m:02}' for m in range(3, 6)]
            data_dir = os.path.join(self.radar_path, f'{year}{months[0]}01T0000_to_{year}{months[-1]}30T2359')
        elif season in ['fall', 'autumn']:
            months = [f'{m:02}' for m in range(8, 12)]
            data_dir = os.path.join(self.radar_path, f'{year}{months[0]}01T0000_to_{year}{months[-1]}30T2359')
        elif season == 'test':
            months = ['08']
            data_dir = os.path.join(self.radar_path, f'{year}0801T0000_to_{year}1130T2359')

        radars = datahandling.load_radars(data_dir)

        lons, lats = zip(*list(radars.values()))
        bounds = [max(lats)+1, min(lons)-1, min(lats)-1, max(lons)+1] # North, West, South, East. Default: global

        if season == 'test':
            days = ['01']
        else:
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

        self.pressure_level_config.update(info)
        self.surface_data_config.update(info)

        os.makedirs(target_dir, exist_ok=True)
        file_path = os.path.join(target_dir, filename)
        #self.client.retrieve('reanalysis-era5-land',
        #                     self.surface_data_config,
        #                     file_path)

        self.client.retrieve('reanalysis-era5-pressure-levels',
                             self.pressure_level_config,
                             file_path)

        return radars, file_path


def extract_points(data_path, lonlat_list, t_range):
    data = xr.open_dataset(data_path)
    data = data.rio.write_crs('EPSG:4326') # set crs to lat lon
    data = data.rio.interpolate_na() # fill nan's by interpolating spatially

    weather = {}

    for var, ds in data.items():
        var_data = []
        for lonlat in lonlat_list:
            # Extract time-series data at given point (interpolate between available grid points)
            data_point = ds.interp(longitude=lonlat[0], latitude=lonlat[1], method='linear')
            var_data.append(data_point.sel(time=t_range).data.flatten())
        weather[var] = np.stack(var_data)

    return weather



if __name__ == '__main__':

    loader = ERA5Loader()
    base_dir = '/home/fiona/environmental_data/era5'
    for year in ['2015']: #'2015', '2016', '2017', '2018']:
        for season in ['spring', 'fall']:
            radars, file_path = loader.download_season(year,
                                                       season,
                                                       os.path.join(base_dir, season, year),
                                                       f'wind_850.nc')