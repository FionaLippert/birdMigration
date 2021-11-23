import numpy as np
import os
import os.path as osp
import pandas as pd
import geopandas as gpd
import pickle5 as pickle
from yaml import Loader, load
import itertools as it
import networkx as nx
import scipy.stats as stats
from matplotlib import pyplot as plt
import seaborn as sb
from matplotlib import cm
from shapely import geometry
import geoplot as gplt
import torch
from cartopy.feature import ShapelyFeature
from matplotlib.ticker import FixedLocator
import cartopy.crs as ccrs


def load_cv_results(result_dir, ext='', trials=1):

    result_list = []
    for t in range(1, trials+1):
        file = osp.join(result_dir, f'trial_{t}', f'results{ext}.csv')
        if osp.isfile(file):
            df = pd.read_csv(file)
            df['trial'] = t
            result_list.append(df)

            cfg_file = osp.join(result_dir, f'trial_{t}', 'config.yaml')
            with open(cfg_file) as f:
                cfg = load(f, Loader=Loader)

    results = pd.concat(result_list)

    return results, cfg

def load_model_fluxes(result_dir, ext='', trials=1):

    fluxes = {}
    for t in range(1, trials + 1):
        file = osp.join(result_dir, f'trial_{t}', f'model_fluxes{ext}.pickle')

        with open(file, 'rb') as f:
            fluxes[t] = pickle.load(f)

    return fluxes



def flux_corr_per_dist2boundary(voronoi, model_fluxes, gt_fluxes):

    # shortest paths to any boundary cell
    sp = nx.shortest_path(G)
    d_to_b = np.zeros(len(G))
    for ni, datai in G.nodes(data=True):
        min_d = np.inf
        for nj, dataj in G.nodes(data=True):
            if dataj['boundary']:
                d = len(sp[ni][nj])
                if d < min_d:
                    min_d = d
                    d_to_b[ni] = d
    voronoi['dist2boundary'] = d_to_b

    print(voronoi)

    df = dict(radar1=[], radar2=[], corr=[], dist2boundary=[])
    for i, rowi in voronoi.iterrows():
        for j, rowj in voronoi.iterrows():
            if not i == j:
                df['radar1'].append(rowi['radar'])
                df['radar2'].append(rowj['radar'])
                df['dist2boundary'].append(int(min(rowi['dist2boundary'], rowj['dist2boundary'])))
                if not np.all(model_fluxes[i, j] == 0) and not np.all(gt_fluxes[i, j] == 0) and np.all(
                        np.isfinite(gt_fluxes[i, j])):
                    df['corr'].append(stats.pearsonr(gt_fluxes[i, j].flatten(), model_fluxes[i, j].flatten())[0])
                else:
                    df['corr'].append(np.nan)
    df = pd.DataFrame(df)

    return df

def fluxes_per_dist2boundary(G, voronoi):
    
    # shortest paths to any boundary cell
    sp = nx.shortest_path(G)
    d2b = np.zeros(len(G))
    for ni, datai in G.nodes(data=True):
        min_d = np.inf
        for nj, dataj in G.nodes(data=True):
            if dataj['boundary']:
                d = len(sp[ni][nj])
                if d < min_d:
                    min_d = d
                    d2b[ni] = d
    voronoi['dist2boundary'] = d2b

    d2b_index = {}
    for i, j in G.edges():
        d2b = int(voronoi.iloc[i]['dist2boundary'])
        if not d2b in d2b_index.keys():
            d2b_index[d2b] = dict(idx=[], jdx=[])
        
        d2b_index[d2b]['idx'].append(i)
        d2b_index[d2b]['jdx'].append(j)

    return d2b_index

def fluxes_per_angle(G, bins=12):
    angle_index = {}
    bins = np.linspace(0, 360, bins+1)
    binc = (bins[1:] + bins[:-1]) / 2
    for i, j, data in G.edges(data=True):
        angle = (data['angle'] + 360) % 360
        angle_bin = np.where(bins < angle)[0][-1]
        angle_bin = binc[angle_bin]

        if not angle_bin in angle_index.keys():
            angle_index[angle_bin] = dict(idx=[], jdx=[])

        angle_index[angle_bin]['idx'].append(i)
        angle_index[angle_bin]['jdx'].append(j)

    return angle_index


def flux_corr_per_angle(G, model_fluxes, gt_fluxes):
    df = dict(radar1=[], radar2=[], corr=[], angle=[])
    for i, j, data in G.edges(data=True):
        df['radar1'].append(G.nodes(data=True)[i]['radar'])
        df['radar2'].append(G.nodes(data=True)[j]['radar'])
        df['angle'].append((G.get_edge_data(i, j)['angle'] % 360))
        if not np.all(model_fluxes[i, j] == 0) and not np.all(gt_fluxes[i, j] == 0) and np.all(
                np.isfinite(gt_fluxes[i, j])):
            df['corr'].append(stats.pearsonr(gt_fluxes[i, j].flatten(), model_fluxes[i, j].flatten())[0])
        else:
            df['corr'].append(np.nan)
    df = pd.DataFrame(df)

    return df

def bin_metrics_fluxes(model_fluxes, gt_fluxes):
    mask = np.logical_and(np.isfinite(gt_fluxes), gt_fluxes != 0)

    model_bin = model_fluxes[mask].flatten() > 0
    gt_bin = gt_fluxes[mask].flatten() > 0

    tp = np.logical_and(model_bin, gt_bin).sum()
    fp = np.logical_and(model_bin, ~gt_bin).sum()
    fn = np.logical_and(~model_bin, gt_bin).sum()
    tn = np.logical_and(~model_bin, ~gt_bin).sum()

    summary = dict(precision = [], sensitivity = [], accuracy = [])
    summary['precision'].append(tp / (tp + fp))
    summary['sensitivity'].append(tp / (tp + fn))
    summary['accuracy'].append((tp + tn) / (tp + fp + tn + fn))

    return summary
    



def plot_fluxes(voronoi, extent, G, fluxes, bird_scale=1,
                ax=None, crs=None, max_flux=None, cbar=True):
    G_new = nx.DiGraph()
    G_new.add_nodes_from(list(G.nodes(data=True)))

    radars = voronoi.radar.values

    for i, ri in enumerate(radars):
        for j, rj in enumerate(radars):

            val = np.nanmean(fluxes[j, i])

            if val > 0 and i != j:
                boundary1 = ('boundary' in ri) and ('boundary' in rj)
                boundary2 = voronoi.query(f'radar == "{ri}" or radar == "{rj}"')['boundary'].all()

                if not boundary1 and not boundary2:
                    G_new.add_edge(j, i, flux=val)

    coord_df = gpd.GeoDataFrame(dict(radar=voronoi.radar,
                                     observed=voronoi.observed,
                                     geometry=[geometry.Point((row.lon, row.lat)) for i, row in voronoi.iterrows()]),
                                crs='epsg:4326').to_crs(crs)
    pos = {ridx: (
    coord_df.query(f'radar == "{name}"').geometry.iloc[0].x, coord_df.query(f'radar == "{name}"').geometry.iloc[0].y)
           for
           (ridx, name) in nx.get_node_attributes(G_new, 'radar').items()}

    fluxes = np.array(list(nx.get_edge_attributes(G_new, 'flux').values()))
    fluxes *= bird_scale
    if max_flux is None:
        max_flux = fluxes.max()

    c_radar = 'lightgray'
    c_marker = '#0a3142'
    cmap = cm.get_cmap('YlOrRd')
    norm = plt.Normalize(0, max_flux)
    edge_colors = cmap(norm(fluxes))
    edge_widths = np.minimum(fluxes, max_flux) / (0.25 * max_flux) + 0.8

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    nx.draw(G_new, pos=pos, with_labels=False, ax=ax,
            node_size=9000 / len(G_new), node_color=c_radar, width=edge_widths,
            connectionstyle="arc3,rad=0.1", edge_color=edge_colors)#,
            #options={'arrowsize': edge_widths*100}, zorder=2)

    gplt.polyplot(coord_df.query('observed == 1').buffer(20_000).to_crs(epsg=4326),
                  ax=ax, extent=extent, zorder=3, edgecolor=c_marker, linewidth=1.5)
    gplt.polyplot(coord_df.query('observed == 0').buffer(20_000).to_crs(epsg=4326),
                  ax=ax, extent=extent, zorder=3, edgecolor=c_marker, linewidth=2)
    gplt.pointplot(coord_df.query('observed == 1').to_crs(epsg=4326),
                  ax=ax, extent=extent, zorder=4, edgecolor=c_marker, s=6)
    if cbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm, extend='max').set_label(label='number of birds', size=15);

    return G_new, ax

if __name__ == "__main__":

    models = { 'FluxGraphLSTM': ['final_evaluation_64'] }

    trials = 5
    year = 2017
    season = 'fall'
    H = 24

    ext = ''
    datasource = 'abm'
    n_dummy = 25

    base_dir = '/home/flipper/birdMigration'
    result_dir = osp.join(base_dir, 'results', datasource)
    # data_dir = osp.join(base_dir, 'data', 'raw', 'abm', season, str(year))
    data_dir = osp.join(base_dir, 'data', 'preprocessed', f'1H_voronoi_ndummy={n_dummy}',
                        datasource, season, str(year))

    dep = np.load(osp.join(data_dir, 'departing_birds.npy'))
    land = np.load(osp.join(data_dir, 'landing_birds.npy'))
    delta = dep - land

    with open(osp.join(data_dir, 'time.pkl'), 'rb') as f:
        abm_time = pickle.load(f)
    time_dict = {t: idx for idx, t in enumerate(abm_time)}


    voronoi = gpd.read_file(osp.join(base_dir, 'data', 'preprocessed',
                                     f'1H_voronoi_ndummy={n_dummy}',
                                     'abm', season, str(year), 'voronoi.shp'))

    radar_dict = voronoi.radar.to_dict()
    radar_dict = {v: k for k, v in radar_dict.items()}

    #inner_radars = voronoi.query('boundary == 0').radar.values
    #boundary_idx = voronoi.query('boundary == 1').index.values

    G = nx.read_gpickle(osp.join(base_dir, 'data', 'preprocessed', f'1H_voronoi_ndummy={n_dummy}',
                                 'abm', season, str(year), 'delaunay.gpickle'))


    def get_abm_data(data, datetime, radar, bird_scale=1):
        tidx = time_dict[pd.Timestamp(datetime)]
        ridx = radar_dict[radar]
        return data[tidx, ridx] / bird_scale

    inner_radars = voronoi.query('observed == 1').radar.values
    boundary_idx = voronoi.query('observed == 0').index.values

    gt_fluxes = np.load(osp.join(data_dir, 'outfluxes.npy'))


    for m, dirs in models.items():
        print(f'evaluate model components for {m}')
        for d in dirs:
            result_dir = osp.join(base_dir, 'results', datasource, m, f'test_{year}', d)
            results, cfg = load_cv_results(result_dir, ext=ext, trials=trials)
            model_fluxes = load_model_fluxes(result_dir, ext=ext, trials=trials)
            bird_scale = cfg['datasource']['bird_scale']
            output_dir = osp.join(result_dir, 'performance_evaluation', f'H={H}')
            os.makedirs(output_dir, exist_ok=True)

            area_scale = results.area.max()

            #df = results.query(f'horizon == {H}')
            df = results.query(f'horizon <= 24')
            df = df[df.radar.isin(inner_radars)]
            df['month'] = pd.DatetimeIndex(df.datetime).month

            print('evaluate source/sink')

            corr_source = dict(month=[], trial=[], corr=[])
            corr_sink = dict(month=[], trial=[], corr=[])

            for m in df.month.unique():
                print(f'evaluate month {m}')
                for t in df.trial.unique():
                    data = df.query(f'month == {m} & trial == {t}')

                    print('compute abm source/sink')
                    data['gt_source_km2'] = data.apply(
                       lambda row: get_abm_data(dep, row.datetime, row.radar) / (row.area / area_scale), axis=1)
                    data['gt_sink_km2'] = data.apply(
                           lambda row: get_abm_data(land, row.datetime, row.radar) / (row.area / area_scale), axis=1)

                    print('aggregate source/sink over 24 H')
                    grouped = data.groupby(['seqID', 'radar'])
                    grouped = grouped[['gt_source_km2', 'source_km2', 'gt_sink_km2', 'sink_km2']].aggregate(
                        np.nansum).reset_index()

                    print('compute correlation')
                    corr = np.corrcoef(grouped.gt_source_km2.to_numpy(),
                                grouped.source_km2.to_numpy())[0, 1]
                    corr_source['month'].append(m)
                    corr_source['trial'].append(t)
                    corr_source['corr'].append(corr)

                    corr = np.corrcoef(grouped.gt_sink_km2.to_numpy(),
                                       grouped.sink_km2.to_numpy())[0, 1]
                    corr_sink['month'].append(m)
                    corr_sink['trial'].append(t)
                    corr_sink['corr'].append(corr)

            corr_source = pd.DataFrame(corr_source)
            corr_sink = pd.DataFrame(corr_sink)

            corr_source.to_csv(osp.join(output_dir, 'agg_source_corr_per_trial.csv'))
            corr_sink.to_csv(osp.join(output_dir, 'agg_sink_corr_per_trial.csv'))




            #df['gt_source_km2'] = df.apply(
            #    lambda row: get_abm_data(dep, row.datetime, row.radar) / (row.area / area_scale), axis=1)
            #df['gt_sink_km2'] = df.apply(
            #        lambda row: get_abm_data(land, row.datetime, row.radar) / (row.area / area_scale), axis=1)

            # corr per radar
            #gr = results[results.radar.isin(inner_radars)].dropna().groupby(['radar', 'trial'])
            #corr = gr[['abm_delta', 'source/sink']].corr().iloc[0::2, -1].reset_index()
            #corr.to_csv(osp.join(output_dir, f'delta_corr_per_radar{ext}.csv'))

            # corr per gt bin
            
            #gr = results[results.radar.isin(inner_radars)].dropna().groupby(['seqID', 'radar', 'trial'])
            
            #activity = gr['gt'].aggregate(np.nanmean).reset_index()
            #results['activity_bin'] = pd.cut(results['gt_km2'] / birdscale, bins=np.linspace(0, 1, 4))
            
            """
            df['month'] = pd.DatetimeIndex(df.datetime).month

            grouped = df.groupby(['seqID', 'radar', 'trial', 'month'])
            grouped = grouped[['gt_source_km2', 'source_km2', 'gt_sink_km2', 'sink_km2']].aggregate(np.nansum).reset_index()
            #grouped = df.groupby(['trial', 'month'])
            corr_source = grouped[['gt_source_km2', 'source_km2']].corr().iloc[0::2, -1].reset_index()
            corr_source.to_csv(osp.join(output_dir, 'agg_source_corr_per_trial.csv'))

            corr_sink = grouped[['gt_sink_km2', 'sink_km2']].corr().iloc[0::2, -1].reset_index()
            corr_sink.to_csv(osp.join(output_dir, 'agg_sink_corr_per_trial.csv'))
            """

            #corr = gr[['abm_delta', 'source/sink']].corr().iloc[0::2, -1].reset_index()
            #joint = corr.join(activity, how='outer', rsuffix='_r')
            #joint['activity_bin'] = pd.cut(joint['gt'].values, bins=np.arange(0, joint['gt'].max()+200, 200))
            #joint.to_csv(osp.join(output_dir, f'delta_corr_per_activity_bin{ext}.csv'))

            # corr per hour
            #gr = results[results.radar.isin(inner_radars)].dropna().groupby(['horizon', 'trial'])
            #corr = gr[['abm_delta', 'source/sink']].corr().iloc[0::2, -1].reset_index()
            #corr.to_csv(osp.join(output_dir, f'delta_corr_per_hour{ext}.csv'))

            context = cfg['model']['context']
            horizon = cfg['model']['test_horizon']

            # rearange abm fluxes to match model fluxes
            gt_fluxes_H = []
            gt_times = []
            for s in sorted(results.groupby('seqID').groups.keys()):
                df = results.query(f'seqID == {s}')
                time = sorted(df.datetime.unique())
                t = time[context + H]
                gt_times.append(t)
                gt_fluxes_H.append(gt_fluxes[time_dict[pd.Timestamp(t)]])

            #context = cfg['model']['context']
            #horizon = cfg['model']['test_horizon']
            # gt_flux = np.stack([f[..., context: context + horizon] for
            #             f in gt_flux_dict.values()], axis=-1)
            gt_fluxes = np.stack(gt_fluxes_H, axis=-1)
            gt_times = np.stack(gt_times, axis=-1)

            # exclude "self-fluxes"
            for i in range(gt_fluxes.shape[0]):
                gt_fluxes[i, i] = np.nan

            # exclude boundary to boundary fluxes
            for i, j in it.product(boundary_idx, repeat=2):
                gt_fluxes[i, j] = np.nan

            # aggregate fluxes per sequence
            # gt_flux_per_seq = gt_flux.sum(2)

            # net fluxes
            gt_net_fluxes = gt_fluxes - np.moveaxis(gt_fluxes, 0, 1)
            # gt_net_flux_per_seq = gt_flux_per_seq - np.moveaxis(gt_flux_per_seq, 0, 1)

            #gt_net_fluxes = gt_net_fluxes[..., :5]

            overall_corr = {}
            corr_per_radar = {}
            corr_per_hour = {}
            corr_d2b = []
            corr_angles = []
            bin_fluxes = []
            # loop over all trials
            for t, model_fluxes_t in model_fluxes.items():

                print(f'evaluate fluxes for trial {t}')
                seqIDs = sorted(model_fluxes_t.keys())
                model_fluxes_t = np.stack([model_fluxes_t[s].detach().numpy()[..., H] for s in seqIDs], axis=-1)
                model_net_fluxes_t = model_fluxes_t - np.moveaxis(model_fluxes_t, 0, 1)
                # model_flux_per_seq_t = model_flux_per_seq_t = model_flux_t.sum(2)
                # model_net_flux_per_seq_t = model_flux_per_seq_t - np.moveaxis(model_flux_per_seq_t, 0, 1)
                #model_net_fluxes_t = model_net_fluxes_t[..., :5]

                mask = np.isfinite(gt_net_fluxes)
                overall_corr[t] = np.corrcoef(gt_net_fluxes[mask].flatten(),
                                              model_net_fluxes_t[mask].flatten())[0, 1]

                
                bin_results = bin_metrics_fluxes(model_net_fluxes_t, gt_net_fluxes)
                bin_results = pd.DataFrame(bin_results)
                bin_results['trial'] = t
                bin_fluxes.append(bin_results)

                d2b_index = fluxes_per_dist2boundary(G, voronoi)
                corr_per_d2b = dict(d2b=[], corr=[], trial=[])
                for d2b, index in d2b_index.items():
                    model_net_fluxes_d2b = model_net_fluxes_t[index['idx'], index['jdx']]
                    gt_net_fluxes_d2b = gt_net_fluxes[index['idx'], index['jdx']]
                    mask = np.isfinite(gt_net_fluxes_d2b)
                    corr = stats.pearsonr(model_net_fluxes_d2b[mask].flatten(), gt_net_fluxes_d2b[mask].flatten())[0]
                    corr_per_d2b['d2b'].append(d2b)
                    corr_per_d2b['corr'].append(corr)
                    corr_per_d2b['trial'].append(t)
                corr_d2b.append(pd.DataFrame(corr_per_d2b))


                angle_index = fluxes_per_angle(G)
                corr_per_angle = dict(angle=[], rad=[], corr=[], trial=[])
                for angle, index in angle_index.items():
                    model_net_fluxes_a = model_net_fluxes_t[index['idx'], index['jdx']]
                    gt_net_fluxes_a = gt_net_fluxes[index['idx'], index['jdx']]
                    mask = np.isfinite(gt_net_fluxes_a)
                    corr = stats.pearsonr(model_net_fluxes_a[mask].flatten(), gt_net_fluxes_a[mask].flatten())[0]
                    
                    corr_per_angle['angle'].append(angle)
                    corr_per_angle['rad'].append(angle / 360 * 2 * np.pi)
                    corr_per_angle['corr'].append(corr)
                    corr_per_angle['trial'].append(t)
                corr_angles.append(pd.DataFrame(corr_per_angle))


                #df_corr_d2b = flux_corr_per_dist2boundary(voronoi, model_net_fluxes_t, gt_net_fluxes)
                #df_corr_d2b.to_csv(osp.join(output_dir, f'flux_corr_d2b_{t}.csv'))
                #df_corr_d2b['trial'] = t
                #corr_d2b.append(df_corr_d2b)


                #fig, ax = plt.subplots(figsize=(4, 4))
                #sb.boxplot(x='dist2boundary', y='corr', data=df_corr_d2b.dropna(), ax=ax, width=0.6, linewidth=2, color='lightgray')
                #ax.set_xlabel('distance to boundary', fontsize=12)
                #ax.set_ylabel('correlation coefficient', fontsize=12)
                #plt.grid(color='gray', linestyle='--', alpha=0.5);
                #fig.savefig(osp.join(output_dir, f'flux_corr_d2b_{t}.png'), bbox_inches='tight', dpi=200)



                #df_corr_angles = flux_corr_per_angle(G, model_net_fluxes_t, gt_net_fluxes)
                #df_corr_angles.to_csv(osp.join(output_dir, f'flux_corr_angles_{t}.csv'))
                #df_corr_angles['trial'] = t

                #bins = np.linspace(0, 360, 13)
                #df_corr_angles['angle_bin'] = pd.cut(df_corr_angles['angle'], bins)
                #df_corr_angles['angle_bin'] = df_corr_angles['angle_bin'].apply(lambda deg: (deg.left + deg.right) / 2)
                #df_corr_angles['rad_bin'] = df_corr_angles['angle_bin'].apply(lambda deg: deg / 360 * 2 * np.pi)
                #corr_angles.append(df_corr_angles)

                #fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'projection': 'polar'})
                #grouped = df_corr_angles.groupby('rad_bin')
                #means = grouped.aggregate(np.nanmean).reset_index()
                #stds = grouped.aggregate(np.nanstd).reset_index()
                #bars = ax.bar(means['rad_bin'], means['corr'], width=0.3, bottom=0,
                #              yerr=stds['corr'], ecolor='black', color='gray')
                
                #max_corr = means['corr'].max() + stds['corr'].max()
                #ax.set_rlim(0, max_corr)
                #ax.set_theta_zero_location("N")
                #ax.set_theta_direction(-1)
                #fig.savefig(osp.join(output_dir, f'flux_corr_angles_{t}.png'), bbox_inches='tight', dpi=200)


                # plot map with avg fluxes
                shape_dir = osp.join(base_dir, 'data', 'shapes')
                countries = gpd.read_file(osp.join(shape_dir, 'ne_10m_admin_0_countries_lakes.shp'))
                bounds = voronoi.total_bounds
                clon = np.mean([bounds[0], bounds[2]])
                clat = np.mean([bounds[1], bounds[3]])
                extent = [-6, 16, 41, 56]
                crs = ccrs.AlbersEqualArea(central_longitude=clon, central_latitude=clat)

                fig, ax = plt.subplots(figsize=(14, 10), subplot_kw={'projection': crs})
                ax.set_extent(extent)
                f = ShapelyFeature(countries.geometry, ccrs.PlateCarree(), edgecolor='white')
                ax.add_feature(f, facecolor='lightgray', zorder=0)

                G_model, ax = plot_fluxes(voronoi, extent, G, model_net_fluxes_t,
                            ax=ax, crs=crs.proj4_init, max_flux=8)

                gl = ax.gridlines(crs=ccrs.PlateCarree(),
                                  linewidth=1, color='gray', alpha=0.25, linestyle='--')
                xspacing = FixedLocator([-10, -5, 0, 5, 10, 15, 20])
                yspacing = FixedLocator([40, 45, 50, 55])
                gl.xlocator = xspacing
                gl.ylocator = yspacing
                ax.set_extent(extent)
                fig.savefig(osp.join(output_dir, f'flux_map_{t}.png'), bbox_inches='tight', dpi=200)
                nx.write_gpickle(G_model, osp.join(output_dir, f'model_fluxes_{t}.gpickle'), protocol=4)

                if t == 1:
                    G_gt, ax = plot_fluxes(voronoi, extent, G, gt_net_fluxes, crs=crs.proj4_init, max_flux=8)
                    nx.write_gpickle(G_gt, osp.join(output_dir, 'gt_fluxes.gpickle'), protocol=4)
            
            corr_d2b = pd.concat(corr_d2b)
            corr_angles = pd.concat(corr_angles)
            bin_fluxes = pd.concat(bin_fluxes)
            corr_d2b.to_csv(osp.join(output_dir, 'corr_d2b_per_trial.csv'))
            corr_angles.to_csv(osp.join(output_dir, 'corr_angles_per_trial.csv'))
            bin_fluxes.to_csv(osp.join(output_dir, 'bins_per_trial.csv'))

            with open(osp.join(output_dir, 'overall_corr.pickle'), 'wb') as f:
                pickle.dump(overall_corr, f, pickle.HIGHEST_PROTOCOL)
            #
            # with open(osp.join(output_dir, 'corr_per_radar.pickle'), 'wb') as f:
            #     pickle.dump(corr_per_radar, f, pickle.HIGHEST_PROTOCOL)
            #
            # with open(osp.join(output_dir, 'corr_per_hour.pickle'), 'wb') as f:
            #     pickle.dump(corr_per_hour, f, pickle.HIGHEST_PROTOCOL)







