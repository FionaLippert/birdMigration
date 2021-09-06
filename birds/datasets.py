#import torch
#from torch_geometric.data import Data, DataLoader, Dataset, InMemoryDataset
from omegaconf import DictConfig, OmegaConf
import numpy as np
import networkx as nx
import os.path as osp
import os
import pandas as pd
import geopandas as gpd
import pickle5 as pickle
from pvlib import solarposition
import itertools as it

from birds import spatial, datahandling, era5interface, abm


def static_features(data_dir, year, **kwargs):

    season = kwargs.get('season', 'fall')
    radar_dir = osp.join(data_dir, 'radar', season, year)
    radars = datahandling.load_radars(radar_dir)
    radars = {k: v for k, v in radars.items() if not v in kwargs.get('exclude', [])}

    # voronoi tesselation and associated graph
    space = spatial.Spatial(radars, n_dummy_radars=kwargs.get('n_dummy_radars', 0))
    voronoi, G = space.voronoi()
    # G = space.subgraph(G, 'type', 'measured')  # graph without sink nodes
    #G_max_dist = space.G_max_dist(kwargs.get('max_distance', 250))

    print('create radar buffer dataframe')
    # 25 km buffers around radars


    radar_buffers = gpd.GeoDataFrame({'radar': voronoi.radar,
                                     'observed' : voronoi.observed},
                                     geometry=space.pts_local.buffer(25_000),
                                     crs=space.crs_local)
    print(space.pts_local.crs)
    print(radar_buffers.crs)
    #radar_buffers.set_crs(space.crs_local)

    # compute areas of voronoi cells and radar buffers [unit is km^2]
    radar_buffers['area_km2'] = radar_buffers.area / 10**6
    voronoi['area_km2'] = voronoi.area / 10**6

    radar_buffers = radar_buffers.to_crs(f'epsg:{space.epsg_lonlat}')
    voronoi = voronoi.to_crs(f'epsg:{space.epsg_lonlat}')

    print('done with static preprocessing')

    return voronoi, radar_buffers, G #, G_max_dist

def dynamic_features(data_dir, year, data_source, voronoi, radar_buffers, **kwargs):

    env_points = kwargs.get('env_points', 100)
    season = kwargs.get('season', 'fall')
    random_seed = kwargs.get('seed', 1234)
    pref_dirs = kwargs.get('pref_dirs', {'spring': 58, 'fall': 223})
    pref_dir = pref_dirs[season]
    wp_threshold = kwargs.get('wp_threshold', -0.5)
    edge_type = kwargs.get('edge_type', 'voronoi')
    t_unit = kwargs.get('t_unit', '1H')

    print(f'##### load data for {season} {year} #####')

    if data_source == 'radar':
        print(f'load radar data')
        radar_dir = osp.join(data_dir, 'radar')
        voronoi_radars = voronoi.query('observed == True')
        birds_km2, _, t_range = datahandling.load_season(radar_dir, season, year, ['vid'],
                                                         t_unit=t_unit, mask_days=False,
                                                         radar_names=voronoi_radars.radar,
                                                         interpolate_nans=False)

        radar_data, _, t_range = datahandling.load_season(radar_dir, season, year, ['ff', 'dd', 'u', 'v'],
                                                          t_unit=t_unit, mask_days=False,
                                                          radar_names=voronoi_radars.radar,
                                                          interpolate_nans=True)

        bird_speed = radar_data[:, 0, :]
        bird_direction = radar_data[:, 1, :]
        bird_u = radar_data[:, 2, :]
        bird_v = radar_data[:, 3, :]

        data = birds_km2 * voronoi_radars.area_km2.to_numpy()[:, None] # rescale according to voronoi cell size
        t_range = t_range.tz_localize('UTC')

    elif data_source == 'abm':
        print(f'load abm data')
        abm_dir = osp.join(data_dir, 'abm')
        voronoi_radars = voronoi.query('observed == True')
        print(voronoi_radars.radar.values)
        radar_buffers_radars = radar_buffers.query('observed == True')
        print(radar_buffers_radars.crs)
        data, t_range, bird_u, bird_v = abm.load_season(abm_dir, season, year, voronoi_radars)
        print('loaded voronoi cell data')
        buffer_data = abm.load_season(abm_dir, season, year, radar_buffers_radars, uv=False)[0]
        print('loaded buffer data')

        birds_km2 = buffer_data / radar_buffers_radars.area_km2.to_numpy()[:, None] # rescale to birds per km^2
        buffer_data = birds_km2 * voronoi_radars.area_km2.to_numpy()[:, None] # rescale to birds per voronoi cell

    # time range for solar positions to be able to infer dusk and dawn
    solar_t_range = t_range.insert(-1, t_range[-1] + pd.Timedelta(t_range.freq))
    #solar_t_range = solar_t_range.insert(0, t_range[0] - pd.Timedelta(t_range.freq))

    print('load env data')
    env_vars = ['u', 'v', 'u10', 'v10', 'cc', 'tp', 'sp', 't2m', 'sshf']

    if edge_type == 'voronoi':
        env_areas = voronoi.geometry
    else:
        env_areas = radar_buffers.geometry
    env_850 = era5interface.compute_cell_avg(osp.join(data_dir, 'env', season, year, 'pressure_level_850.nc'),
                                         env_areas, env_points,
                                         t_range.tz_localize(None), vars=env_vars, seed=random_seed)
    env_surface = era5interface.compute_cell_avg(osp.join(data_dir, 'env', season, year, 'surface.nc'),
                                         env_areas, env_points,
                                         t_range.tz_localize(None), vars=env_vars, seed=random_seed)

    dfs = []
    for ridx, row in voronoi.iterrows():

        df = {}

        df['radar'] = [row.radar] * len(t_range)

        # time related variables for radar ridx
        solarpos = np.array(solarposition.get_solarposition(solar_t_range, row.lat, row.lon).elevation)
        night = np.logical_or(solarpos[:-1] < -6, solarpos[1:] < -6)
        df['solarpos_dt'] = solarpos[:-1] - solarpos[1:]
        df['solarpos'] = solarpos[:-1]
        df['night'] = night
        df['dusk'] = np.logical_and(solarpos[:-1] >=6, solarpos[1:] < 6)  # switching from day to night
        df['dawn'] = np.logical_and(solarpos[:-1] < 6, solarpos[1:] >=6)  # switching from night to day
        df['datetime'] = t_range
        df['dayofyear'] = pd.DatetimeIndex(t_range).dayofyear
        df['tidx'] = np.arange(t_range.size)

        # bird measurements for radar ridx
        df['birds'] = data[ridx] if row.observed else [np.nan] * len(t_range)
        df['birds_km2'] = birds_km2[ridx] if row.observed else [np.nan] * len(t_range)

        cols = ['birds', 'birds_km2', 'bird_u', 'bird_v']

        df['bird_u'] = bird_u[ridx] if row.observed else [np.nan] * len(t_range)
        df['bird_v'] = bird_v[ridx] if row.observed else [np.nan] * len(t_range)

        if data_source == 'abm':
            df['birds_from_buffer'] = buffer_data[ridx] if row.observed else [np.nan] * len(t_range)
            cols.append('birds_from_buffer')
        else:
            df['birds_from_buffer'] = data[ridx] if row.observed else [np.nan] * len(t_range)
            df['bird_speed'] = bird_speed[ridx] if row.observed else [np.nan] * len(t_range)
            df['bird_direction'] = bird_direction[ridx] if row.observed else [np.nan] * len(t_range)

            cols.extend(['birds_from_buffer', 'bird_speed', 'bird_direction'])


        # environmental variables for radar ridx
        for var in env_vars:
            if var in env_850:
                print(f'found {var} in env_850 dataset')
                df[var] = env_850[var][ridx]
            elif var in env_surface:
                print(f'found {var} in surface dataset')
                df[var] = env_surface[var][ridx]
        df['wind_speed'] = np.sqrt(np.square(df['u']) + np.square(df['v']))
        # Note that here wind direction is the direction into which the wind is blowing,
        # which is the opposite of the standard meteorological wind direction

        df['wind_dir'] = (abm.uv2deg(df['u'], df['v']) + 360) % 360

        # compute accumulation variables (for baseline models)
        groups = [list(g) for k, g in it.groupby(enumerate(df['night']), key=lambda x: x[-1])]
        nights = [[item[0] for item in g] for g in groups if g[0][1]]
        df['nightID'] = np.zeros(t_range.size)
        df['acc_rain'] = np.zeros(t_range.size)
        df['acc_wind'] = np.zeros(t_range.size)
        df['wind_profit'] = np.zeros(t_range.size)
        acc_rain = 0
        u_rain = 0
        acc_wind = 0
        u_wind = 0
        for nidx, night in enumerate(nights):
            df['nightID'][night] = np.ones(len(night)) * (nidx + 1)

            # accumulation due to rain in the past nights
            acc_rain = acc_rain/3 + u_rain * 2/3
            df['acc_rain'][night] = np.ones(len(night)) * acc_rain
            # compute proportion of hours with rain during the night
            u_rain = np.mean(df['tp'][night] > 0.01)

            # accumulation due to unfavourable wind in the past nights
            acc_wind = acc_wind/3 + u_wind * 2/3
            df['acc_wind'][night] = np.ones(len(night)) * acc_wind
            # compute wind profit for bird with speed 12 m/s and flight direction 223 degree north
            v_air = np.ones(len(night)) * 12
            alpha = np.ones(len(night)) * pref_dir
            df['wind_profit'][night] = v_air - np.sqrt(v_air**2 + df['wind_speed'][night]**2 -
                                                       2 * v_air * df['wind_speed'][night] *
                                                       np.cos(np.deg2rad(alpha-df['wind_dir'][night])))
            u_wind = np.mean(df['wind_profit'][night]) < wp_threshold

        radar_df = pd.DataFrame(df)

        # remember missing bird density observations
        radar_df['missing'] = radar_df['birds'].isna()
        print(f'number of missing data points = {radar_df.missing.sum()}')

        for col in cols:
            # set bird quantities to 0 during the day
            radar_df[col] = radar_df[col] * radar_df['night']
            # fill missing bird measurements by interpolation
            if col == 'bird_direction':
                # use "nearest", to avoid artifacts of interpolating between e.g. 350 and 2 degree
                radar_df[col] = radar_df[col].interpolate(method='nearest')
            else:
                # for all other quantities simply interpolate linearly
                radar_df[col] = radar_df[col].interpolate(method='linear')

        dfs.append(radar_df)

    dynamic_feature_df = pd.concat(dfs, ignore_index=True)
    print(f'columns: {dynamic_feature_df.columns}')
    return dynamic_feature_df



def prepare_features(target_dir, data_dir, year, data_source, **kwargs):

    radar_years = kwargs.get('radar_years', ['2015', '2016', '2017'])
    process_dynamic = kwargs.get('process_dynamic', True)

    # load static features
    if data_source == 'abm' and not year in radar_years:
        radar_year = radar_years[-1]
    else:
        radar_year = year
    voronoi, radar_buffers, G = static_features(data_dir, radar_year, **kwargs)

    # save to disk
    #TODO fix fiona error that occurs when saving voronoi GeoSeries
    voronoi.to_file(osp.join(target_dir, 'voronoi.shp'))
    static_feature_df = voronoi.drop(columns='geometry')
    print('save static features')
    static_feature_df.to_csv(osp.join(target_dir, 'static_features.csv'))
    print('save radar buffers')
    radar_buffers.to_file(osp.join(target_dir, 'radar_buffers.shp'))
    nx.write_gpickle(G, osp.join(target_dir, 'delaunay.gpickle'), protocol=4)

    if process_dynamic:
        # load dynamic features
        dynamic_feature_df = dynamic_features(data_dir, year, data_source, voronoi, radar_buffers, **kwargs)
        # save to disk
        dynamic_feature_df.to_csv(osp.join(target_dir, 'dynamic_features.csv'))
