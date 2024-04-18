import cdsapi
from cdo import Cdo
import os
import os.path as osp
import glob
import xarray as xr
import rioxarray
import numpy as np
import pandas as pd
import geopandas as gpd



class ERA5Loader():
    """
    Wrapper for easy loading of ERA5 environmental data for given area and season.
    """

    def __init__(self, **kwargs):
        """
        Initialization of variables to download
        """

        self.single_level_config = kwargs.get('single_level_config', None)
        self.pressure_level_config = kwargs.get('pressure_level_config', None)
        self.model_level_config = kwargs.get('model_level_config', None)

        self.client = cdsapi.Client()


    def download_season(self, season, year, target_dir, **kwargs):
        """
        Download environmental variables for the given year and season.

        :param season: season of interest ('spring' or 'fall')
        :param year: year of interest
        :param target_dir: directory to write downloaded data to
        """

        if season == 'spring':
            months = [f'{m:02}' for m in range(3, 7)]
            levels = '135/127/115' # corresponds to 54m, 334m, 1329m for the ICAO Standard Atmosphere
        else:
            months = [f'{m:02}' for m in range(8, 12)]
            levels = '135/128/115' # corresponds to 54m, 288m, 1329m for the ICAO Standard Atmosphere
        level_names = ['q10', 'q50', 'q90']

        print(season, months)

        # datetime is interpreted as 00:00 UTC
        days = [f'{(d + 1):02}' for d in range(31)]
        time = [f'{h:02}:00' for h in range(24)]

        info = { 'year' : year,
                 'month' : months,
                 'day' : days,
                 'time' : time }

        os.makedirs(target_dir, exist_ok=True)

        if self.single_level_config is not None:
            self.single_level_config.update(info)
            self.client.retrieve('reanalysis-era5-single-levels',
                                self.single_level_config,
                                osp.join(target_dir, 'surface.nc'))

        if self.pressure_level_config is not None:
            self.pressure_level_config.update(info)
            self.client.retrieve('reanalysis-era5-pressure-levels',
                                 self.pressure_level_config,
                                 osp.join(target_dir, f'pressure_levels.nc'))

        if self.model_level_config is not None:
            # model levels are downloaded from MARS tape, and thus need to be loaded month by month
            for month in months:
                info = {'date': f'{year}-{month}-01/to/{year}-{month}-31',
                        'time': '00/to/23/by/1',
                        'levelist': levels
                        }
                self.model_level_config.update(info)

                self.client.retrieve('reanalysis-era5-complete',
                                     self.model_level_config,
                                     osp.join(target_dir, f'{year}_{month}_model_levels.nc'))

            # use CDO to merge all files of the same year
            cdo = Cdo()
            filenames = glob.glob(osp.join(target_dir, f'{year}_*_model_levels.nc'))
            combined_file = osp.join(target_dir, f'model_levels_combined.nc')
            final_file = osp.join(target_dir, f'model_levels.nc')

            cdo.mergetime(input=' '.join(filenames), output=combined_file, options='-b F64')

            # remove level dimension and instead include level info in variable name
            level_mapping = dict(zip(levls.split('/'), level_names))
            prepare_model_level_data(combined_file, final_file, level_mapping)

            os.remove(combined_file)

            # remove individual files after merge
            for item in os.listdir(target_dir):
                if item.startswith(year):
                    os.remove(osp.join(target_dir, item))



def extract_points(data_path, lons, lats, t_range, vars='all'):
    """
    Open downloaded data and extract data at the given coordinates (interpolate if necessary)

    :param data_path: path to downloaded data
    :param lons: list of longitude values
    :param lats: list of latitude values
    :param t_range: time range (in UTC)
    :param vars: list of environmental variables to extract
    :return: numpy.array with shape [number of coordinates, number of variables]
    """

    data = xr.open_dataset(data_path)
    data = data.rio.write_crs('EPSG:4326') # set crs to lat lon
    data = data.rio.interpolate_na() # fill nan's by interpolating spatially

    vars = data.data_vars if vars == 'all' else vars

    lons = xr.DataArray(lons, dims='points')
    lats = xr.DataArray(lats, dims='points')

    print(f'available ERA5 variables: {data.keys()}')

    vars = set(data.keys()).intersection(set(vars))

    data_points = data[vars].interp(longitude=lons, latitude=lats, time=t_range, method='linear')

    weather = {}

    for var in vars:
        weather[var] = data_points[var].data

    # weather = {}
    #
    # for var, ds in data.items():
    #     if var in vars:
    #         var_data = []
    #         for lonlat in lonlat_list:
    #             # Extract time-series data at given point (interpolate between available grid points)
    #             data_point = ds.interp(longitude=lonlat[0], latitude=lonlat[1], method='linear')
    #             var_data.append(data_point.sel(time=t_range).data.flatten())
    #         weather[var] = np.stack(var_data)

    # return weather

    return weather

def sample_points_from_polygon(polygon, seed, n_points=1):
    """
    Sample a random point within the given polygon.

    :param polygon: shapely.geometry.Polygon
    :param seed: random seed
    :return: coordinates (lon, lat)
    """

    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = polygon.bounds

    final_lons = []
    final_lats = []

    while len(final_lons) < n_points:
        lons = rng.random.uniform(minx, maxx, n_points)
        lats = rng.random.uniform(miny, maxy, n_points)
        points = gpd.points_from_xy(lons, lats)
        idx = points.within(polygon)
        final_lons.extend(lons[idx])
        final_lats.extend(lats[idx])

    return {'lon': final_lons[:n_points], 'lat': final_lats[:n_points]}


def compute_cell_avg_sampled(data_path, cell_geometries, n_points, t_range, vars='all', seed=1234):
    """
    For all cells, sample a number of points within the cell and compute cell averages of environmental variables.

    :param data_path: path to downloaded data
    :param cell_geometries: geopandas.GeoDataFrame with cell shapes
    :param n_points: number of points to sample
    :param t_range: time range (in UTC, not localized)
    :param vars: list of variables to extract
    :param seed: random seed
    :return: numpy.array with shape [number of cells, number of variables]
    """

    cell_geometries = cell_geometries.to_crs('EPSG:4326')
    data = xr.open_dataset(data_path)
    data = data.rio.write_crs('EPSG:4326') # set crs to lat lon
    data = data.rio.interpolate_na() # fill nan's by interpolating spatially

    vars = data.data_vars if vars == 'all' else vars

    # sample points within cells
    points = {i: sample_points_from_polygon(poly, seed=seed, n_points=n_points)
              for i, poly in enumerate(cell_geometries)}

    weather = {}
    for var, ds in data.items():
        if var in vars:
            var_data = []
            for i, poly in enumerate(cell_geometries):
                var_data_poly = []
                for j in range(n_points):
                    # lon, lat = sample_point_from_polygon(poly, seed=seed)
                    lon = points[i]['lon'][j]
                    lat = points[i]['lat'][j]
                    # Extract time-series data at given point (interpolate between available grid points)
                    interp = ds.interp(longitude=lon, latitude=lat, method='linear')
                    interp = interp.sel(time=t_range).data.flatten()
                    var_data_poly.append(interp)
                var_data.append(np.nanmean(np.stack(var_data_poly, axis=0), axis=0))
            weather[var] = np.stack(var_data, axis=1)

    return weather


def compute_cell_avg(data_path, cell_geometries, t_range, vars='all'):
    """
    For all cells, compute cell averages of environmental variables.

    :param data_path: path to downloaded data
    :param cell_geometries: geopandas.GeoDataFrame with cell shapes
    :param t_range: time range (in UTC, not localized)
    :param vars: list of variables to extract
    :param seed: random seed
    :return: numpy.array with shape [number of cells, number of variables]
    """

    cell_geometries = cell_geometries.to_crs('EPSG:4326')
    data = xr.open_dataset(data_path)
    data = data.rio.write_crs('EPSG:4326') # set crs to lat lon
    #data = data.rio.interpolate_na() # fill nan's by interpolating spatially

    #data = adjust_coords(data, coord='longitude')
    #data = adjust_coords(data, coord='latitude')

    #tidx = data.time.isin(t_range)
    #data = data.isel(time=tidx)
    data = data.drop_duplicates('time')
    data = data.sel(time=t_range)

    vars = data.data_vars if vars == 'all' else vars

    weather = {}
    for var, ds in data.items():
        if var in vars:
            # select time period
            #tidx = ds.time.isin(t_range)
            #ds = ds.isel(time=tidx)
            print(ds.shape, len(t_range))
            # compute cell averages
            
            weather[var] = []
            for cell in cell_geometries:
                weather[var].append(ds.rio.clip([cell], cell_geometries.crs).mean(dim=['latitude', 'longitude']).values)
            weather[var] = np.stack(weather[var], axis=1)
            #weather[var] = np.stack([ds.rio.clip([cell], cell_geometries.crs).mean(dim=['latitude', 'longitude']).values
            #                         for cell in cell_geometries], axis=1) # shape [time, cells]

    return weather

def prepare_model_level_data(input_file, output_file, level_mapping={}):

    data = xr.open_dataset(input_file)

    data = flatten_levels(data, level_mapping)
    data = adjust_coords(data, 'longitude')
    data = adjust_coords(data, 'latitude')
    data = data.drop_duplicates(dim='time')
    
    data.to_netcdf(output_file)


def flatten_levels(data, level_mapping):

    ds_list = []
    for var, ds in data.items():
        if 'level' in ds.coords:
            ds = ds.to_dataset(dim='level')
            var_mapping = {l: f'{var}_{level_mapping[str(l)] if str(l) in level_mapping else f"L{l}"}' for l in ds.data_vars}
            ds = ds.rename(var_mapping)
        ds_list.append(ds)

    data = xr.merge(ds_list)

    return data

def adjust_coords(ds, coord='longitude'):

    # Adjust lon values to make sure they are within (-180, 180)
    ds[f'_{coord}_adjusted'] = xr.where(ds[coord] > 180, ds[coord] - 360, ds[coord])

    # reassign the new coords to as the main lon coords
    # and sort DataArray using new coordinate values
    ds = (ds
        .swap_dims({coord: f'_{coord}_adjusted'})
        .sel(**{f'_{coord}_adjusted': sorted(ds[f'_{coord}_adjusted'])})
        .drop(coord))

    ds = ds.rename({f'_{coord}_adjusted': coord})

    return ds
