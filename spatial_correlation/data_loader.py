import numpy as np
import xarray as xr
import warnings
import pandas as pd
import glob, os
from matplotlib import pyplot as plt

def add_coords_to_xr(f):
    ds = xr.open_dataset(f)
    ds = ds.expand_dims({'latitude': [ds.latitude],
                         'longitude': [ds.longitude]})
    return ds


def get_coords(f):
    ds = xr.open_dataset(f)
    return (ds.longitude, ds.latitude)


def get_name(f):
    ds = xr.open_dataset(f)
    return ds.source


def to_latlon(coords):
    return (coords[1], coords[0])


def resample(f, start, end, var, unit, mask_days):
    ds = xr.open_dataset(f)
    if mask_days:
        # only keep data at times between civil dusk and dawn (6 degrees below horizon)
        ds = ds.where(ds.solarpos < -6)
    ds = ds.sel(time=slice(start,end))[var]
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='Mean of empty slice')
        ds = ds.resample(time=unit, skipna=True).reduce(np.nanmean)
    t_range = pd.date_range(start, end, freq=unit)
    ds = ds.reindex({'time': t_range})
    return ds


def load_data(path, var='vid', start=None, end=None, t_unit='1H', mask_days=True):
    files = glob.glob(os.path.join(path,'*.nc'))
    data = {get_coords(f) : resample(f, start, end, var, t_unit, mask_days) for f in files}
    names = {get_coords(f): get_name(f) for f in files}
    t_range = pd.date_range(start, end, freq=t_unit)
    return data, names, t_range


def arr(xarray_1d):
    return np.array(xarray_1d).flatten()


def plot_all_vpts(data, names, t_range, bar=False):
    # data, names, t_range = load_data(path, var, start, end, t_unit)

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