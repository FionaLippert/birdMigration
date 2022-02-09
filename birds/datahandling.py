import numpy as np
import xarray as xr
import warnings
import pandas as pd
import glob, os
import os.path as osp
from matplotlib import pyplot as plt
from pvlib import solarposition

#########################################################################################################
# This is a collection of methods for interfacing with VID (vertically integrated density)
# weather radar data in netCDF format. For each radar and season, there should be a separate file.
#########################################################################################################


def load_season(root, season, year, vars=['vid'], t_unit='1H', mask_days=True, radar_names=[], interpolate_nans=False):
    """
    Load all data available for the given year and season.

    :param root: directory containing all raw data for a particular datasource
    :param season: season of interest ('spring' or 'fall')
    :param year: year of interest
    :param vars: variables to extract from radar files
    :param t_unit: time unit (data is resampled to have uniform time resolution)
    :param mask_days: only consider time points during the night
    :param radar_names: list of radar names to consider
    :param interpolate_nans: fill missing data by interpolation
    :return: data (mapping from radar names to measurements), radar names (mapping from coordinates to names), time range
    """

    path = os.path.join(root, season, year)

    if season == 'spring':
        start = f'{year}-03-15 12:00:00'
        end = f'{year}-05-15 12:00:00'

    elif season == 'fall':
        start = f'{year}-08-01 12:00:00'
        end = f'{year}-11-15 12:00:00'

    dataset, radars, t_range = load_data(path, vars, start, end, t_unit, mask_days, interpolate_nans)

    if len(radar_names) == 0:
        radar_names = radars.values()
    data = np.stack([dataset[radar].to_array().squeeze() for radar in radar_names], axis=0)

    return data, radars, t_range


def load_data(path, vars=['vid'], start=None, end=None, t_unit='1H', mask_days=True, interpolate_nans=False):
    """
    Load data for all radars available in 'path'.

    :param path: path to directory containing netcdf files (*.nc) for all radars
    :param vars: variables to extract from radar files
    :param start: start datetime (string)
    :param end: end datetime (string)
    :param t_unit: time unit (data is resampled to have uniform time resolution)
    :param mask_days: only consider time points during the night
    :param interpolate_nans: fill missing data by interpolation
    :return: data (mapping from radar names to measurements), radar names (mapping from coordinates to names), time range
    """

    files = glob.glob(os.path.join(path, '*.nc'))
    data = {get_name(f): resample(f, start, end, vars, t_unit, mask_days, interpolate_nans) for f in files}
    names = {get_coords(f): get_name(f) for f in files}
    t_range = pd.date_range(start, end, freq=t_unit)
    return data, names, t_range


def load_radars(path):
    """
    Find all radars for which data is available in the given path.

    :param path: path to directory containing netcdf files (*.nc)
    :return: available radars (mapping from coordinates to radar names)
    """
    files = glob.glob(os.path.join(path, '*.nc'))
    radars = {get_coords(f): get_name(f) for f in files}
    return radars


def get_solarpos(t_range_utc, lonlat_pairs):
    """
    Load solar positions for the given time range and coordinates.

    :param t_range_utc: time range (in UTC)
    :param lonlat_pairs: list of coordinates [(lon1, lat1), ....]
    :return: numpy.ndarray with shape [number of coordinates, number of time points]
    """

    solarpos = [solarposition.get_solarposition(t_range_utc, lat, lon).elevation for lon, lat in lonlat_pairs]
    solarpos = np.stack(solarpos, axis=0)
    return solarpos


def resample(f, start, end, vars, unit, mask_days, interpolate_nans):
    """Resample radar data along the time dimension"""

    ds = xr.open_dataset(f)
    if mask_days:
        # only keep data at times between civil dusk and dawn (6 degrees below horizon)
        ds = ds.where(ds.solarpos < -6)
    ds = ds.sel(time=slice(start, end))[vars]
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='Mean of empty slice')
        ds = ds.resample(time=unit, base=0, skipna=True).reduce(np.nanmean)
    t_range = pd.date_range(start, end, freq=unit)
    ds = ds.reindex({'time': t_range})
    if interpolate_nans:
        ds = ds.interpolate_na(dim='time', method="linear")
    return ds

def add_coords_to_xr(f):
    """Add info about longitude and latitude of measurements to xarray dataset"""

    ds = xr.open_dataset(f)
    ds = ds.expand_dims({'latitude': [ds.latitude],
                         'longitude': [ds.longitude]})
    return ds


def get_coords(f):
    """Load dataset and extract coodinates (lon, lat) from it"""

    ds = xr.open_dataset(f)
    if hasattr(ds, 'longitude') and hasattr(ds, 'latitude'):
        return (ds.longitude, ds.latitude)
    else:
        return(ds.lon.values[0], ds.lat.values[0])


def get_name(f, opera_format=True):
    """Load dataset and extract radar name from it"""

    ds = xr.open_dataset(f)
    if hasattr(ds, 'source'):
        radar = ds.source
    else:
        filename = osp.splitext(osp.basename(f))[0]
        radar = filename.split('_')[-1]

    if opera_format:
        if '/' in radar:
            radar = (radar[:2]+radar[-3:]).lower()
    else:
        if radar.islower():
            radar = radar[:2] + '/' + radar[-3:]

    return radar


def to_latlon(coords):
    """Flip coordinates from (lon, lat) to (lat, lon)"""
    return (coords[1], coords[0])


def arr(xarray_1d):
    """Return xarray variable as numpy.array"""
    return xarray_1d.values.flatten()


def plot_all_vpts(data, names, t_range, bar=False):
    """Plot time series data for all radars"""

    fig, ax = plt.subplots(len(data), 1, figsize=(15, 2 * len(data)), sharex=True)
    for i, (coord, ds) in enumerate(data.items()):
        ds_arr = arr(ds)

        if bar:
            barwidth = 0.95 * (1 / t_range.shape[0])
            ax[i].bar(t_range, ds_arr, barwidth)
        else:
            ax[i].plot(t_range, ds_arr)

        missing = (np.where(np.isnan(ds_arr))[0].size / ds_arr.size) * 100

        ax[i].text(*(.98, .95), f'{names[coord]} [{missing:.0f}% missing]',
                   horizontalalignment='right', va="top",
                   transform=ax[i].transAxes, fontsize=12)
    plt.subplots_adjust(hspace=0)
