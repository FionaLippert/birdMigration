import geoplot as gplt
from matplotlib import pyplot as plt
import geoplot.crs as gcrs
import cartopy.crs as ccrs
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import os.path as osp
import glob
import pickle5 as pickle


def load_sim_results(path):
    files = glob.glob(osp.join(path, '*.pkl'))
    traj = []
    states = []
    for file in files:
        with open(file, 'rb') as f:
            result = pickle.load(f)
            traj.append(result['trajectories'])
            states.append(result['states'])

    traj = np.concatenate(traj, axis=1)
    states = np.concatenate(states, axis=1)
    time = result['time']
    return traj, states, time

def background_map(ax, countries, departure_area, extent=[0.36, 46.36, 16.07, 55.40]):

    ax = gplt.polyplot(
        countries,
        edgecolor="white",
        facecolor="lightgray",
        ax=ax,
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


def draw_birds(ax, countries, departure_area, traj, states, anim_path, tidx, time):
    ax, extent = background_map(ax, countries, departure_area)

    flying = np.where(states == 1)
    ground = np.where(states == 0)
    if len(flying[0]) > 0:
        xx = traj[flying, 0].flatten()
        yy = traj[flying, 1].flatten()
        df = gpd.GeoSeries(gpd.points_from_xy(xx, yy, crs='epsg:4326'))
        gplt.pointplot(df, ax=ax, extent=extent, zorder=1, color='red', alpha=0.8)
        # points = [Point(x, y) for x,y in zip(xx, yy)]
        # gplt.pointplot(gpd.GeoSeries(points, crs='epsg:4326'), ax=ax, extent=extent, zorder=1, color='red', alpha=0.8)
    if len(ground[0]) > 0:
        xx = traj[ground, 0].flatten()
        yy = traj[ground, 1].flatten()
        points = [Point(x, y) for x, y in zip(xx, yy)]
        gplt.pointplot(gpd.GeoSeries(points, crs='epsg:4326'), ax=ax, extent=extent, zorder=2, color='blue', alpha=0.2)

    ax.set_title(time)
    file_name = osp.join(anim_path, f'{tidx}.png')
    plt.savefig(file_name, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    return file_name

def plot_trajectories(ax, countries, departure_area, traj, states, birds=[0]):
    ax, extent = background_map(ax, countries, departure_area)
    for bidx in birds:
        xx = traj[:, bidx, 0]
        yy = traj[:, bidx, 1]
        tr = LineString([Point(x, y) for x, y in zip(xx, yy)])
        start = Point(xx[0], yy[0])
        stopovers = [Point(x, y) for t, (x, y) in enumerate(zip(xx, yy)) if states[t, bidx] == 0]
        gplt.polyplot(gpd.GeoSeries(tr, crs='epsg:4326'), ax=ax, zorder=2, extent=extent)
        gplt.pointplot(gpd.GeoSeries(stopovers, crs='epsg:4326'), ax=ax, zorder=3, extent=extent, edgecolor='black',
                       color='white', linewidth=0.5)
        gplt.pointplot(gpd.GeoSeries(start, crs='epsg:4326'), ax=ax, zorder=3, extent=extent, color='red')

    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')


def make_grid(extent=[0.36, 46.36, 16.07, 55.40], res=0.5, crs='4326'):
    xmin, ymin, xmax, ymax = extent
    cols = np.arange(int(np.floor(xmin))-1, int(np.ceil(xmax))+1, res)
    rows = np.arange(int(np.floor(ymin))-1, int(np.ceil(ymax))+1, res)
    rows = rows[::-1]
    polygons = []
    for x in cols:
        for y in rows:
            polygons.append(Polygon([(x,y), (x+res, y), (x+res, y-res), (x, y-res)]))

    grid = gpd.GeoDataFrame({'geometry': polygons}, crs=f'epsg:{crs}')
    return grid

def get_points(trajectories, states, state=1):
    df = gpd.GeoDataFrame({'geometry': []}, crs='epsg:4326')
    mask = np.where(states == state)
    if len(mask[0]) > 0:
        xx = trajectories[mask, 0].flatten()
        yy = trajectories[mask, 1].flatten()
        df['geometry'] = gpd.points_from_xy(xx, yy)
    return df


def aggregate(trajectories, states, grid, t_range, state):
    names = []
    grid_counts = grid.to_crs('epsg:4326')
    for t in t_range:
        merged = gpd.sjoin(get_points(trajectories[t], states[t], state), grid_counts, how='left', op='within')
        merged[f'n_birds_{t}'] = 1
        dissolve = merged.dissolve(by="index_right", aggfunc="count")
        name_t = f'n_birds_{t}'
        grid_counts.loc[dissolve.index, name_t] = dissolve[name_t].values
        names.append(name_t)
    return grid_counts, names
