#!/usr/bin/env python
import cdsapi

months = [f'{m:02}' for m in range(8, 12)]
days   = [f'{(d+1):02}' for d in range(31)]
hours  = [f'{h:02}:00' for h in range(24)]

c = cdsapi.Client()
c.retrieve('reanalysis-era5-pressure-levels', {
        'variable'      : 'temperature',
        'pressure_level': '1000',
        'product_type'  : 'reanalysis',
        'year'          : '2016',
        'month'         : '09',
        'day'           : days,
        'area'          : [57.94, -1.94, 44.91, 10.49], # North, West, South, East. Default: global
        'grid'          : [0.1, 0.1], # Latitude/longitude grid: east-west (longitude) and north-south resolution (latitude). Default: 0.25 x 0.25
        'time'          : hours,
        'format'        : 'netcdf' # Supported format: grib and netcdf. Default: grib
    }, 'era5_temperatures.nc')
