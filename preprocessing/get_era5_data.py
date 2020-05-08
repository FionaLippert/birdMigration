#!/usr/bin/env python
import cdsapi
import os

years  = ['2016']
months = [f'{m:02}' for m in range(8, 12)]
days   = [f'{(d+1):02}' for d in range(31)]
hours  = [f'{h:02}:00' for h in range(24)]

bounds =  [54.9533386, 2.7999701, 50.9533386, 6.7999701] #[54, -1, 46, 7] # North, West, South, East. Default: global
resolution = [0.25, 0.25]

target_dir = 'weather_data'

c = cdsapi.Client()
c.retrieve('reanalysis-era5-pressure-levels', {
        'variable'      : [
                #'fraction_of_cloud_cover',
                #'specific_humidity',
                'specific_rain_water_content',
                'temperature',
                'u_component_of_wind',
                'v_component_of_wind',
        ],
        'pressure_level': [
                '800',
                '900',
                '1000',
        ],
        'product_type'  : 'reanalysis',
        'year'          : years,
        'month'         : months,
        'day'           : days,
        'area'          : bounds,
        'grid'          : resolution, # Latitude/longitude grid: east-west (longitude) and north-south resolution (latitude). Default: 0.25 x 0.25
        'time'          : hours,
        'format'        : 'netcdf' # Supported format: grib and netcdf. Default: grib
    }, os.path.join(target_dir, 'era5_pressure_levels.nc'))


c.retrieve('reanalysis-era5-land', {
        'variable'      : [
                #'2m_temperature',
                'surface_sensible_heat_flux',
                #'10m_u_component_of_wind',
                #'10m_v_component_of_wind'
                'surface_pressure',
                'total_precipitation',
        ],
        'year'          : years,
        'month'         : months,
        'day'           : days,
        'area'          : bounds,
        'grid'          : resolution, # Latitude/longitude grid: east-west (longitude) and north-south resolution (latitude). Default: 0.25 x 0.25
        'time'          : hours,
        'format'        : 'netcdf' # Supported format: grib and netcdf. Default: grib
    }, os.path.join(target_dir, 'era5_land.nc'))
