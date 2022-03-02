# birdMigration

A collection of methods for 
- loading and preprocessing weather radar observations of bird movements
- loading environmental data from ERA5
- simulating nocturnal bird migration with an agent/indiviual based model.

## Getting started
- the R package `uvaRadar` is used to download weather radar data (requires login)
- the Python package `cdsapi` is used to download ERA5 reanalysis data (requires login; steps are described [here](https://cds.climate.copernicus.eu/api-how-to))
- before accessing h5 data using `wradlib` run `export WRADLIB_DATA=~/`

## Weather radar data loading
To load weather radar data from the database, first switch to the `dataloading` directory, 
specify the radars and times of interest in the `config.yml` file, and then run
```
python run_vpts_generation.py path/to/output_dir
```

## ERA5 data loading
To load environmental data for the geographical extent covered by weather radars, first switch to the 
`dataloading` directory, and then run
```
python load_era5.py --root path/to/data_dir --years 2015 2016 2017
```

## Simulating nocturnal bird migration
To simulate bird trajectories in parallel, switch to the `simulation` directory,
and then run
```
python parallel_simulation.py --root path/to/data_dir
```
