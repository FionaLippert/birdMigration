#!/usr/bin/env python
import cdsapi
import os
import argparse
from ../../python/modules import datahandling

"""
NL/HRW: lon 5.1381, lat 51.8369
NL/DHL: lon 4.79997, lat 52.95334
BE/JAB: lon 3.0642, lat 51.1917 
BE/ZAV: lon 4.455, lat 50.9055
"""


parser = argparse.ArgumentParser(description='download era5 data for a given location.')
#parser.add_argument('lat', type=float, help='latitude')
#parser.add_argument('lon', type=float, help='longitude')
parser.add_argument('year', type=float, help='year')
parser.add_argument('radar', type=str, help='radar station')
args = parser.parse_args()

radar_locs = {
        "NL/HRW": (5.1381, 51.8369),
        "NL/DHL": (4.79997, 52.95334),
        "BE/JAB": (3.0642, 51.1917),
        "BE/ZAV": (4.455, 50.9055)
}

path = '/home/fiona/radar_data/vpi/night_only'
data_path = os.path.join(path, f'{args.year}0801T0000_to_{args.year}1130T2359')
datahandling.load_radars()

years  = [args.year]
months = [f'{m:02}' for m in range(8, 12)]
days   = [f'{(d+1):02}' for d in range(31)]
time   = [f'{h:02}:{m:02}' for h in range(24) for m in range(0, 59, 15)]



#bounds =  [args.lat, args.lon, args.lat, args.lon] #[54, -1, 46, 7] # North, West, South, East. Default: global
lon, lat = radar_locs[args.radar]
bounds =  [lat, lon, lat, lon]
#resolution = [0.25, 0.25]

target_dir = 'weather_data'
rad = f'{args.radar.split("/")[0]}{args.radar.split("/")[1]}'

c = cdsapi.Client()
pressure_level_data = c.retrieve('reanalysis-era5-pressure-levels', {
        'variable'      : [
                #'fraction_of_cloud_cover',
                'specific_humidity',
                #'specific_rain_water_content',
                'temperature',
                'u_component_of_wind',
                'v_component_of_wind',
        ],
        'pressure_level': [
                #'400',
                '500',
                #'600',
                #'700',
                #'800',
                #'900',
                '1000',
        ],
        'product_type'  : 'reanalysis',
        'year'          : years,
        'month'         : months,
        'day'           : days,
        'area'          : bounds,
        #'grid'          : resolution, # Latitude/longitude grid: east-west (longitude) and north-south resolution (latitude). Default: 0.25 x 0.25
        'time'          : time,
        'format'        : 'netcdf' # Supported format: grib and netcdf. Default: grib
    }, os.path.join(target_dir, f'era5_pressure_levels_{rad}.nc'))


surface_level_data = c.retrieve('reanalysis-era5-land', {
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
        #'grid'          : resolution, # Latitude/longitude grid: east-west (longitude) and north-south resolution (latitude). Default: 0.25 x 0.25
        'time'          : time,
        'format'        : 'netcdf' # Supported format: grib and netcdf. Default: grib
    }, os.path.join(target_dir, f'era5_land_{rad}.nc'))


# Extract time-series data at a specific point
data_point = ct.geo.extract_point(
        surface_level_data,
        lat=lat,
        lon=lon
)
print(data_point)