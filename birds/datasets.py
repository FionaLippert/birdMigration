import numpy as np
import networkx as nx
import os.path as osp
import os
import pandas as pd
import geopandas as gpd
import pickle
from pvlib import solarposition
import itertools as it

from birds import spatial, datahandling, era5interface, abm

RADAR_REPLACEMENTS = {'nldbl': 'nlhrw'}


def prepare_features(target_dir, data_dir, year, data_source, **kwargs):
    """
    Prepare static and dynamic features for all radars available in the given year and season.

    :param target_dir: directory where features will be written to
    :param data_dir: directory containting all relevant data
    :param year: year of interest
    :param data_source: 'radar' or 'abm' (simulated data)
    """

    # load static features
    if data_source == 'abm':
        df = gpd.read_file(osp.join(data_dir, 'abm', 'all_radars.shp'))
        radars = dict(zip(zip(df.lon, df.lat), df.radar.values))
    else:
        season = kwargs.get('season', 'fall')
        radar_dir = osp.join(data_dir, 'radar', season, year)
        radars = datahandling.load_radars(radar_dir)

    voronoi, radar_buffers, G = static_features(radars, **kwargs)

    # save to disk
    voronoi.to_file(osp.join(target_dir, 'voronoi.shp'))
    static_feature_df = voronoi.drop(columns='geometry')
    static_feature_df.to_csv(osp.join(target_dir, 'static_features.csv'))
    radar_buffers.to_file(osp.join(target_dir, 'radar_buffers.shp'))
    nx.write_gpickle(G, osp.join(target_dir, 'delaunay.gpickle'), protocol=4)

    if kwargs.get('process_dynamic', True):
        # load dynamic features
        dynamic_feature_df = dynamic_features(data_dir, year, data_source, voronoi, radar_buffers, **kwargs)
        # save to disk
        dynamic_feature_df.to_csv(osp.join(target_dir, 'dynamic_features.csv'))


def static_features(radars, **kwargs):
    """
    For a given set of radars, construct Voronoi tessellation and associated static features.

    These features include: Delaunay triangulation (graph), Voronoi cell areas, length and orientation of Voronoi faces,
    distances between radars, and 25km buffers around radars used for bird estimation.

    :param radars: mapping from radar coordinates (lon, lat) to names
    :return: Voronoi tessellation (geopandas dataframe), radar buffers (geopandas dataframe),
        Delaunay triangulation (networkx graph)
    """

    # check for radars to exclude
    exclude = []
    exclude.extend(kwargs.get('exclude', []))

    for r1, r2 in RADAR_REPLACEMENTS.items():
        # if two radars are available for the same location, use the first one
        if (r1 in radars.values()) and (r2 in radars.values()):
            exclude.append(r2)

    radars = {k: v for k, v in radars.items() if not v in exclude}
    print(f'excluded radars: {exclude}')

    # voronoi tesselation and associated graph
    space = spatial.Spatial(radars, n_dummy_radars=kwargs.get('n_dummy_radars', 0))
    voronoi, G = space.voronoi()

    print('create radar buffer dataframe')
    # 25 km buffers around radars
    radar_buffers = gpd.GeoDataFrame({'radar': voronoi.radar,
                                     'observed' : voronoi.observed},
                                     geometry=space.pts_local.buffer(25_000),
                                     crs=space.crs_local)

    # compute areas of voronoi cells and radar buffers [unit is km^2]
    radar_buffers['area_km2'] = radar_buffers.area / 10**6
    voronoi['area_km2'] = voronoi.area / 10**6

    radar_buffers = radar_buffers.to_crs(f'epsg:{space.epsg_lonlat}')
    voronoi = voronoi.to_crs(f'epsg:{space.epsg_lonlat}')

    print('done with static preprocessing')

    return voronoi, radar_buffers, G

def dynamic_features(data_dir, year, data_source, voronoi, radar_buffers, **kwargs):
    """
    Load all dynamic features, including bird densities and velocities, environmental data, and derived features
    such as estimated accumulation of bird on the ground due to adverse weather.

    Missing data is interpolated, but marked as missing.

    :param data_dir: directory containing all relevant data
    :param year: year of interest
    :param data_source: 'radar' or 'abm' (simulated data)
    :param voronoi: Voronoi tessellation (geopandas dataframe)
    :param radar_buffers: radar buffers with static features (geopandas dataframe)
    :return: dynamic features (pandas dataframe)
    """

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

        # rescale according to voronoi cell size
        data = birds_km2 * voronoi_radars.area_km2.to_numpy()[:, None]
        t_range = t_range.tz_localize('UTC')

    elif data_source == 'abm':
        print(f'load abm data')
        abm_dir = osp.join(data_dir, 'abm')
        voronoi_radars = voronoi.query('observed == True')
        radar_buffers_radars = radar_buffers.query('observed == True')
        data, t_range, bird_u, bird_v = abm.load_season(abm_dir, season, year, voronoi_radars)
        buffer_data = abm.load_season(abm_dir, season, year, radar_buffers_radars, uv=False)[0]

        # rescale to birds per km^2
        birds_km2 = data / voronoi_radars.area_km2.to_numpy()[:, None]
        birds_km2_from_buffer = buffer_data / radar_buffers_radars.area_km2.to_numpy()[:, None]
        # rescale to birds per voronoi cell
        birds_from_buffer = birds_km2_from_buffer * voronoi_radars.area_km2.to_numpy()[:, None]

    # time range for solar positions to be able to infer dusk and dawn
    solar_t_range = t_range.insert(-1, t_range[-1] + pd.Timedelta(t_range.freq))

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

        print(f'preprocess radar {row.radar}')

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

        cols = ['birds', 'birds_km2', 'birds_from_buffer', 'birds_km2_from_buffer', 'bird_u', 'bird_v']

        df['bird_u'] = bird_u[ridx] if row.observed else [np.nan] * len(t_range)
        df['bird_v'] = bird_v[ridx] if row.observed else [np.nan] * len(t_range)

        if data_source == 'abm':
            df['birds_from_buffer'] = birds_from_buffer[ridx] if row.observed else [np.nan] * len(t_range)
            df['birds_km2_from_buffer'] = birds_km2_from_buffer[ridx] if row.observed else [np.nan] * len(t_range)
        else:
            df['birds_from_buffer'] = data[ridx] if row.observed else [np.nan] * len(t_range)
            df['birds_km2_from_buffer'] = birds_km2[ridx] if row.observed else [np.nan] * len(t_range)
            df['bird_speed'] = bird_speed[ridx] if row.observed else [np.nan] * len(t_range)
            df['bird_direction'] = bird_direction[ridx] if row.observed else [np.nan] * len(t_range)

            cols.extend(['bird_speed', 'bird_direction'])


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
        radar_df['missing'] = 0

        for col in cols:
            if data_source == 'radar':
                # radar quantities being exactly 0 during the night are missing,
                # radar quantities during the day are set to 0
                print(f'check missing data for column {col}')
                radar_df[col] = radar_df.apply(lambda row: np.nan if (row.night and not row[col])
                                                            else (0 if not row.night else row[col]), axis=1)

                # remember missing radar observations
                radar_df['missing'] = radar_df['missing'] | radar_df[col].isna()

                # fill missing bird measurements by interpolation
                if col == 'bird_direction':
                    # use "nearest", to avoid artifacts of interpolating between e.g. 350 and 2 degree
                    radar_df[col].interpolate(method='nearest', inplace=True)
                else:
                    # for all other quantities simply interpolate linearly
                    radar_df[col].interpolate(method='linear', inplace=True)
            else:
                radar_df[col] = radar_df.apply(lambda row: np.nan if (row.night and np.isnan(row[col]))
                                                            else (0 if not row.night else row[col]), axis=1)
                radar_df['missing'] = radar_df['missing'] | radar_df[col].isna()

                # fill missing bird measurements with 0
                radar_df[col].fillna(0, inplace=True)

        dfs.append(radar_df)
        print(f'found {radar_df.missing.sum()} misssing time points')

    dynamic_feature_df = pd.concat(dfs, ignore_index=True)
    print(f'columns: {dynamic_feature_df.columns}')
    return dynamic_feature_df
