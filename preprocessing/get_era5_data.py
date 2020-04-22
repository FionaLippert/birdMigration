#!/usr/bin/env python
import cdsapi

c = cdsapi.Client()
c.retrieve('reanalysis-era5-pressure-levels', {
        'variable'      : 'temperature',
        'pressure_level': '1000',
        'product_type'  : 'reanalysis',
        'year'          : '2016',
        'month'         : '10',
        'day'           : '03',
        'area'          : [57.94, -1.94, 44.91, 10.49], # North, West, South, East. Default: global
        'grid'          : [0.1, 0.1], # Latitude/longitude grid: east-west (longitude) and north-south resolution (latitude). Default: 0.25 x 0.25
        'time'          : ['20:00','21:00','22:00','23:00'],
        'format'        : 'netcdf' # Supported format: grib and netcdf. Default: grib
    }, 'era5_temperatures.nc')
