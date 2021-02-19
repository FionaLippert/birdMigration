import numpy as np
from matplotlib import pyplot as plt
import os.path as osp
import os
import pickle5 as pickle
import sys
from functools import partial
import pyproj
from shapely.ops import transform
from shapely.geometry import Point, Polygon

# sys.path.insert(1, osp.join(sys.path[0], '../modules'))
# import abm
# import datahandling
# import spatial

from birds import datahandling, spatial

data_root = '/home/fiona/birdMigration/data'
season = 'fall'
year = '2015'


def bird_counts(data):
    minx = np.min(data['trajectories'][..., 0])
    miny = np.min(data['trajectories'][..., 1])
    maxx = np.max(data['trajectories'][..., 0])
    maxy = np.max(data['trajectories'][..., 1])
    gridx = np.arange(np.ceil(minx), np.ceil(maxx) + 1, 1)
    gridy = np.arange(np.ceil(miny), np.ceil(maxy) + 1, 1)
    counts = np.zeros((data['trajectories'].shape[0], gridx.size, gridy.size))

    for bird in range(data['trajectories'].shape[1]):
        xx = np.digitize(data['trajectories'][:, bird, 0], gridx)
        yy = np.digitize(data['trajectories'][:, bird, 1], gridy)
        fidx = np.where(data['states'][:, bird] == 1)
        for t in fidx[0]:
            counts[t, xx[t], yy[t]] += 1

    return counts




#
# def geodesic_point_buffer(lat, lon, km):
#     # Azimuthal equidistant projection
#     proj_wgs84 = pyproj.Proj('+proj=longlat +datum=WGS84')
#     aeqd_proj = '+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0'
#     project = partial(
#         pyproj.transform,
#         pyproj.Proj(aeqd_proj.format(lat=lat, lon=lon)),
#         proj_wgs84)
#     buf = Point(0, 0).buffer(km * 1000)  # distance in metres
#     return transform(project, buf).exterior.coords[:]



#
#
# radars = datahandling.load_radars(osp.join(data_root, 'raw', 'radar', season, year))
# #print(radars)
# sp = spatial.Spatial(radars)
# buffers = sp.pts_local.buffer(25_000).to_crs(epsg=sp.epsg).to_dict()
#
# radar_counts = {}
# for idx, (lon, lat) in enumerate(radars.keys()):
#     #b1 = Polygon(geodesic_point_buffer(lat, lon, 25))
#     #b2 = buffers[idx]
#     c = np.zeros(data['trajectories'].shape[0])
#
#     for tidx, t in enumerate(data['time']):
#         for bird in range(data['trajectories'].shape[1]):
#             pt = Point(data['trajectories'][tidx, bird])
#             c[tidx] += buffers[idx].contains(pt)
#     radar_counts[(lon, lat)] = c
#
#     fig, ax = plt.subplots()
#     ax.plot(data['time'], c)
#     ax.set_title(radars[(lon, lat)])
#     fig.savefig(osp.join(fig_dir, f'counts_{idx}.png'))
#     plt.close(fig)

# experiment_dir = '/home/fiona/birdMigration/data/experiments/abm/fall/2015/experiment_2021-02-15 15:55:04.073747'
# with open(osp.join(experiment_dir, 'simulation_results_0.pkl'), 'rb') as f:
#     data = pickle.load(f)

# counts = bird_counts(data)
#
# fig, ax = plt.subplots()
# ax.plot(data['time'], counts.reshape((counts.shape[0], -1)).sum(1))
# fig.savefig(osp.join(experiment_dir, 'counts.png'), dpi=200, bbox_inches='tight')
#
# print(f'min lon = {data["trajectories"][:,:,0].min()}')
# print(f'max lon = {data["trajectories"][:,:,0].max()}')
# print(f'min lat = {data["trajectories"][:,:,1].min()}')
# print(f'max lat = {data["trajectories"][:,:,1].max()}')

# num_radars = data['counts'].shape[1]
# for r in range(num_radars):
#     fig, ax = plt.subplots()
#     ax.plot(data['time'], data['counts'][:, r])
#     fig.savefig(osp.join(experiment_dir, f'counts_radar{r}.png'), dpi=200, bbox_inches='tight')
#     plt.close(fig)

# from shapely.ops import cascaded_union
# import geopandas as gpd
# radars = datahandling.load_radars(osp.join(data_root, 'raw', 'radar', season, year))
# sp = spatial.Spatial(radars)
# fig, ax = plt.subplots()
# area = cascaded_union(sp.cells.geometry)
# buff = area.buffer(-10000)
# print(area.area, buff.area)
#
# gpd.GeoSeries(buff).plot(ax=ax, alpha=0.2)
# gpd.GeoSeries(area).plot(ax=ax, alpha=0.2)
# fig.savefig(osp.join(experiment_dir, 'buffer_test.png'), dpi=200)
# plt.close(fig)

import glob
import geopandas as gpd
import cartopy.crs as ccrs
import geoplot.crs as gcrs
import geoplot as gplt

abm_dir = osp.join(data_root, 'raw', 'abm', season, year)
shape_dir = osp.join(data_root, 'shapes')

countries = gpd.read_file(osp.join(shape_dir, 'ne_10m_admin_0_countries_lakes.shp'))
departure_area = gpd.read_file(osp.join(shape_dir, 'departure_area.shp'))
files = glob.glob(os.path.join(abm_dir, '*.pkl'))

data = []
for file in files:
    with open(file, 'rb') as f:
        result = pickle.load(f)
        data.append(result['counts'])

data = np.stack(data, axis=-1).sum(-1)
counts = data.sum(1)

# plot total counts over time
fig, ax = plt.subplots()
ax.plot(result['time'], counts)
fig.savefig(osp.join(abm_dir, 'total_counts.png'), dpi=200, bbox_inches='tight')

fig, ax = plt.subplots(figsize=(12, 8))
gplt.polyplot(
  countries,
  edgecolor="white",
  facecolor="lightgray",
    ax=ax,
  projection=gcrs.AlbersEqualArea(),
  extent=[0.36, 46.36, 16.07, 55.40]
)
gplt.polyplot(
    departure_area,
    facecolor="lightgreen",
    alpha=0.5,
    ax=ax, zorder=1,
    extent=[0.36, 46.36, 16.07, 55.40]
)
ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
fig.savefig(osp.join(abm_dir, 'example_trajectories.png'), dpi=200, bbox_inches='tight')


