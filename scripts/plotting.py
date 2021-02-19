import numpy as np
from matplotlib import pyplot as plt
import os.path as osp
import xarray as xr
import sys
import os
import glob
import pickle5 as pickle
import cartopy.crs as ccrs
import geopandas as gpd
import pandas as pd
import geoplot as gplt
import geoplot.crs as gcrs
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import cascaded_union
import imageio

import birds

root = '/home/fiona/birdMigration/data'
countries = gpd.read_file(osp.join(d, 'ne_10m_admin_0_countries_lakes.shp'))
departure_area = gpd.read_file(osp.join(d, 'departure_area.shp'))
abm_dir = '/home/fiona/birdMigration/data/experiments/abm/fall/2015/experiment_2021-02-17 19:11:34.724467'

files = glob.glob(os.path.join(abm_dir, '*.pkl'))

with open(files[0], 'rb') as f:
    result = pickle.load(f)
    traj = result['trajectories']
    states = result['states']

extent = [0.36, 46.36, 16.07, 55.40]


def background_map():
    ax = gplt.polyplot(
        countries,
        edgecolor="white",
        facecolor="lightgray",
        figsize=(12, 8),
        projection=gcrs.AlbersEqualArea(),
        extent=extent
    )
    ax = gplt.polyplot(
        departure_area,
        facecolor="lightgreen",
        alpha=0.5,
        ax=ax, zorder=1,
        extent=extent
    )
    return ax, extent


def draw_birds(traj, states, output_path, tidx):
    ax, extent = background_map()

    mask = np.where(states)
    print(mask)
    if len(mask[0]) > 0:
        xx = traj[mask, 0].flatten()
        yy = traj[mask, 1].flatten()
        print(xx.shape, yy.shape)
        points = [Point(x, y) for x, y in zip(xx, yy)]
        gplt.pointplot(gpd.GeoSeries(points, crs='epsg:4326'), ax=ax, extent=extent, zorder=1, color='red')

    file_name = osp.join(output_path, f'{i}.png')
    plt.savefig(file_name, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    return file_name

def animate(results, T, output_path):
    files = []
    for tidx in range(T):
        files.append(draw_birds(results['trajectories'][tidx], results['states'][tidx], output_path, tidx))

    images = []
    for f in files:
        images.append(imageio.imread(f))
    gif_path = osp.join(output_path, 'abm_movie.gif')
    imageio.mimsave(gif_path, images, fps=2)


animate(result, 50, abm_dir)