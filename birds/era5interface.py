import cdsapi
import os
import os.path as osp
import xarray as xr
import rioxarray
import numpy as np
from shapely import geometry
import geopandas as gpd

from birds.spatial import Spatial


class ERA5Loader():
    """
    Wrapper for easy loading of ERA5 environmental data for given radar locations.
    """

    def __init__(self, radars):
        """
        Initialization of radars and variables to download.

        :param radars: mapping from coordinates to radar names
        """

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
                                                    'temperature',
                                                    'u_component_of_wind',
                                                    'v_component_of_wind',
                                                    ],
                                    'pressure_level': '850',
                                    'format' : 'netcdf',
                                    'product_type' :'reanalysis',}

        self.client = cdsapi.Client()

    def download_season(self, season, year, target_dir, bounds=None, buffer_x=0, buffer_y=0, pl=850, surface_data=True):
        """
        Download environmental variables for the given year and season.

        :param season: season of interest ('spring' or 'fall')
        :param year: year of interest
        :param target_dir: directory to write downloaded data to
        :param bounds: bounds [North, West, South, East] of geographical area for which data is downloaded (if None, bounds of Voronoi tessellation are used)
        :param buffer_x: buffer around bounds in x-direction (longitude)
        :param buffer_y: buffer around bounds in y-direction (latitude)
        :param pl: pressure level
        :param surface_data: if True, download surface data in addition to pressure level data
        """


        if bounds is None:
            # get bounds of Voronoi tesselation
            spatial = Spatial(self.radars)
            minx, miny, maxx, maxy = spatial.cells.to_crs(epsg=spatial.epsg_lonlat).total_bounds
            bounds = [maxy + buffer_y, minx - buffer_x, miny - buffer_y, maxx + buffer_x]

        if season == 'spring':
            months = [f'{m:02}' for m in range(3, 6)]
        elif season in ['fall', 'autumn']:
            months = [f'{m:02}' for m in range(8, 12)]
        else:
            months = ['09']

        # datetime is interpreted as 00:00 UTC
        days = [f'{(d + 1):02}' for d in range(31)]
        time = [f'{h:02}:00' for h in range(24)]
        resolution = [0.25, 0.25]

        info = { 'year' : year,
                 'month' : months,
                 'day' : days,
                 'area' : bounds,
                 'grid' : resolution,
                 'time' : time }

        os.makedirs(target_dir, exist_ok=True)

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
    """
    Open downloaded data and extract data at the given coordinates (interpolate if necessary)

    :param data_path: path to downloaded data
    :param lonlat_list: list of (lon, lat) coordinates
    :param t_range: time range (in UTC)
    :param vars: list of environmental variables to extract
    :return: numpy.array with shape [number of coordinates, number of variables]
    """

    data = xr.open_dataset(data_path)
    data = data.rio.write_crs('EPSG:4326') # set crs to lat lon
    data = data.rio.interpolate_na() # fill nan's by interpolating spatially

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

def sample_point_from_polygon(polygon, seed):
    """
    Sample a random point within the given polygon.

    :param polygon: shapely.geometry.Polygon
    :param seed: random seed
    :return: coordinates (lon, lat)
    """

    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = polygon.bounds
    lon = np.random.uniform(minx, maxx)
    lat = np.random.uniform(miny, maxy)
    pos = geometry.Point(lon, lat)
    while not polygon.contains(pos):
        lon = rng.uniform(minx, maxx)
        lat = rng.uniform(miny, maxy)
        pos = geometry.Point(lon, lat)
    return lon, lat

def compute_cell_avg(data_path, cell_geometries, n_points, t_range, vars, seed=1234):
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

    weather = {}
    for var, ds in data.items():
        if var in vars:
            var_data = []
            for poly in cell_geometries:
                var_data_poly = []
                for i in range(n_points):
                    lon, lat = sample_point_from_polygon(poly, seed=seed)
                    # Extract time-series data at given point (interpolate between available grid points)
                    interp = ds.interp(longitude=lon, latitude=lat, method='linear')
                    interp = interp.sel(time=t_range).data.flatten()
                    var_data_poly.append(interp)
                var_data.append(np.nanmean(np.stack(var_data_poly, axis=0), axis=0))
            weather[var] = np.stack(var_data, axis=0)

    return weather