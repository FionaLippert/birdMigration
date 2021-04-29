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
import seaborn as sb
import scipy as sp
import pandas as pd

def plot_results_scatter(results, max=1e7, min=0, root_transform=1, legend=False):

    fig, ax = plt.subplots(1, len(results), figsize=(len(results) * 6, 6))
    for midx, m in enumerate(results.keys()):
        gt = results[m]['gt']
        mask = results[m]['night'] & ~results[m]['missing'] & results[m]['gt'] > min
        gt = gt[mask].values
        gt = np.power(gt, 1/root_transform)

        pred = results[m]['prediction']
        pred = pred[mask].values
        pred = np.power(np.maximum(pred, 0), 1/root_transform)

        res = sp.stats.linregress(gt, pred)
        sb.regplot(gt, pred, scatter=True, ci=95, ax=ax[midx], label=f'R-squared={res.rvalue ** 2:.4f}',
                   scatter_kws={'alpha': 0.2, 's': 2})

        ax[midx].plot(np.power([min,max], 1/root_transform), np.power([min, max], 1/root_transform), ls='--', c='red')
        ax[midx].set_title(m)
        if legend: ax[midx].legend()
        ax[midx].set(xlabel='radar observation', ylabel='prediction')
    return fig

def compute_mse(row, bird_scale, prediction_col='prediction'):
    if row['missing']:
        return np.nan
    else:
        return ((row['gt'] - row[prediction_col] * row['night']) / bird_scale) ** 2

def plot_errors(results, bird_scales):
    fig, ax = plt.subplots(figsize=(15, 6))
    for idx, m in enumerate(results.keys()):
        if m == 'GAM':

            results[m]['constant_error'] = results[m].apply(lambda row: compute_mse(row, bird_scales[m],
                                                                                    'constant_prediction'), axis=1)
            mse = results[m].groupby(['horizon', 'trial']).constant_error.mean().apply(np.sqrt)
            mean_mse = mse.groupby('horizon').aggregate(np.mean)
            ax.plot(mean_mse, label='constant prediction')

            end_night = np.concatenate(
                [np.where((mean_mse.iloc[1:] == 0) & (mean_mse.iloc[:-1] > 0))[0], [len(mean_mse)]])
            start_night = np.where((mean_mse.iloc[:-1] == 0) & (mean_mse.iloc[1:] > 0))[0]

            for i, tidx in enumerate(start_night):
                ax.fill_between([tidx + 1, end_night[i]], 0, 0.1, color='lightgray')

        results[m]['error'] = results[m].apply(lambda row: compute_mse(row, bird_scales[m]), axis=1)
        mse = results[m].groupby(['horizon', 'trial']).error.mean().apply(np.sqrt)
        mean_mse = mse.groupby('horizon').aggregate(np.mean)
        std_mse = mse.groupby('horizon').aggregate(np.std)

        l = ax.plot(mean_mse, label=m)
        ax.fill_between(mean_mse.index, mean_mse + std_mse, mean_mse - std_mse, alpha=0.2, color=l[0].get_color())

    plt.legend()
    plt.grid()
    ax.set(xlabel='forecast horizon [h]', ylabel='RMSE', ylim=(0, 0.1))
    return fig

def plot_average_errors(results, bird_scales):
    sb.set(style="ticks")
    fig, ax = plt.subplots(figsize=(20, 4))
    rmse_list = []
    labels = []
    for idx, m in enumerate(results.keys()):
        if m == 'GAM':
            results[m]['constant_error'] = results[m].apply(lambda row: compute_mse(row, bird_scales[m],
                                                                                    'constant_prediction'), axis=1)
            rmse = results[m].groupby(['trial']).constant_error.mean().apply(np.sqrt)
            rmse_list.append(rmse.values)
            labels.append(['constant'] * len(rmse))


        results[m]['error'] = results[m].apply(lambda row: compute_mse(row, bird_scales[m]), axis=1)
        rmse = results[m].groupby(['trial']).error.mean().apply(np.sqrt)
        rmse_list.append(rmse.values)
        labels.append([m] * len(rmse))

    df = pd.DataFrame(dict(RMSE=np.concatenate(rmse_list), model=np.concatenate(labels)))
    sb.barplot(x='model', y='RMSE', data=df, capsize=.2, ci='sd', ax=ax)
    plt.grid()
    ax.set(ylabel='RMSE')
    return fig

def plot_average_errors_comparison(models, results1, results2, bird_scales1, bird_scales2, group_names):
    sb.set(style="ticks")
    fig, ax = plt.subplots(figsize=(10, 4))
    rmse_list = []
    labels = []
    groups = []
    for idx, m in enumerate(models):
        if m == 'GAM':
            results1[m]['constant_prediction_other'] = results2[m]['constant_prediction'] / bird_scales2[m] * bird_scales1[m]
            results1[m]['constant_error'] = results1[m].apply(lambda row: compute_mse(row, bird_scales1[m],
                                                                                    'constant_prediction'), axis=1)
            results1[m]['constant_error_other'] = results1[m].apply(lambda row: compute_mse(row, bird_scales1[m],
                                                                         'constant_prediction_other'), axis=1)
            rmse = results1[m].groupby(['trial']).constant_error.mean().apply(np.sqrt)
            rmse_list.append(rmse.values)
            labels.append(['constant'] * len(rmse))
            groups.append([group_names[0]] * len(rmse))

            rmse = results1[m].groupby(['trial']).constant_error_other.mean().apply(np.sqrt)
            rmse_list.append(rmse.values)
            labels.append(['constant'] * len(rmse))
            groups.append([group_names[1]] * len(rmse))


        results1[m]['error'] = results1[m].apply(lambda row: compute_mse(row, bird_scales1[m]), axis=1)
        rmse = results1[m].groupby(['trial']).error.mean().apply(np.sqrt)
        rmse_list.append(rmse.values)
        labels.append([m] * len(rmse))
        groups.append([group_names[0]] * len(rmse))

        results1[m]['prediction_other'] = results2[m]['prediction'] / bird_scales2[m] * bird_scales1[m]
        results1[m]['error_other'] = results1[m].apply(lambda row: compute_mse(row, bird_scales1[m],
                                                                               'prediction_other'), axis=1)
        rmse = results1[m].groupby(['trial']).error_other.mean().apply(np.sqrt)
        rmse_list.append(rmse.values)
        labels.append([m] * len(rmse))
        groups.append([group_names[1]] * len(rmse))

    df = pd.DataFrame(dict(RMSE=np.concatenate(rmse_list), model=np.concatenate(labels), group=np.concatenate(groups)))
    sb.barplot(x='model', y='RMSE', hue='group', data=df, capsize=.2, ci='sd', ax=ax, palette="Greens_d")
    ax.set(ylabel='RMSE')
    plt.grid()
    return fig

def plot_example_prediction(results, radar, seqID, bird_scales, max=1):

    fig, ax = plt.subplots(figsize=(15, 6))
    for i, m in enumerate(results.keys()):
        r = results[m].query(f'seqID == {seqID} & radar == "{radar}"')
        if i == 0:
            r0 = r.query(f'trial == 0')
            ax.plot(range(len(r0)), r0['gt'] / bird_scales[m], label='radar observation', color='gray')
            #ax.plot(range(len(r0)), r0['constant_prediction'] / bird_scales[m], label='constant')

        all_trials = []
        for trial in r.trial.unique():
            r_t = r.query(f'trial == {trial}')
            all_trials.append(r_t['prediction'] / bird_scales[m]) #* r_t['night'])
        all_trials = np.stack(all_trials, axis=0)

        line = ax.plot(range(all_trials.shape[1]), all_trials.mean(0), label=m)
        ax.fill_between(range(all_trials.shape[1]), all_trials.mean(0) - all_trials.std(0),
                        all_trials.mean(0) + all_trials.std(0), color=line[0].get_color(), alpha=0.2)

    end_night = np.concatenate([np.where(r0['night'].iloc[1:].values & ~r0['night'].iloc[:-1].values)[0], [len(r0)]])
    start_night = np.where(r0['night'].iloc[:-1].values & ~r0['night'].iloc[1:].values)[0]
    for i, tidx in enumerate(start_night):
        ax.fill_between([tidx + 1, end_night[i]], 0, max, color='lightgray')
    ax.set(ylim=(0, max), xlim=(-1, 40), xlabel='forcasting horizon [h]', ylabel='normalized bird density')
    ax.set_title(f'{r0.datetime.values[0]} to {r0.datetime.values[-1]}')
    plt.legend()
    return fig

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

def get_bird_flows(trajectories, states, tidx, grid):
    mask = np.where(states == 1)
    if len(mask[0]) > 0:
        # get grid cell of all flying birds at timestep tidx
        df_t0 = gpd.GeoDataFrame({'geometry': []}, crs='epsg:4326')
        xx_t0 = trajectories[tidx, mask, 0].flatten()
        yy_t0 = trajectories[tidx, mask, 1].flatten()
        df_t0['geometry'] = gpd.points_from_xy(xx_t0, yy_t0)
        merged_t0 = gpd.sjoin(df_t0, grid, how='left', op='within')

        # get grid cell of all previously flying birds at next timestep tidx+1
        df_t1 = gpd.GeoDataFrame({'geometry': []}, crs='epsg:4326')
        xx_t1 = trajectories[tidx+1, mask, 0].flatten()
        yy_t1 = trajectories[tidx+1, mask, 1].flatten()
        df_t1['geometry'] = gpd.points_from_xy(xx_t1, yy_t1)
        merged_t1 = gpd.sjoin(df_t1, grid, how='left', op='within')

        # determine flows
        merged_t0['dst_radar'] = merged_t1['radar']
        merged_t0['dst_index'] = merged_t1['index_right']
        return merged_t0




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

def add_sink_to_voronoi(voronoi, sink):
    gdf_sink = gpd.GeoDataFrame()
    for c in voronoi.columns:
        gdf_sink[c] = [np.nan]
    gdf_sink['radar'] = 'sink'
    gdf_sink['geometry'] = sink.geometry
    voronoi_with_sink = voronoi.append(gdf_sink, ignore_index=True)
    return voronoi_with_sink
