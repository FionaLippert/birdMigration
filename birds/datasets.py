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

# RADAR_REPLACEMENTS = {'nldbl': 'nlhrw'}


def prepare_features(target_dir, data_dir, year, data_source, **kwargs):
    """
    Prepare static and dynamic features for all radars available in the given year and season.

    :param target_dir: directory where features will be written to
    :param data_dir: directory containting all relevant data
    :param year: year of interest
    :param data_source: 'radar' or 'abm' (simulated data)
    """
    print(target_dir)

    # load static features
    if data_source == 'abm':
        print('abm')
        df = gpd.read_file(osp.join(data_dir, 'abm', 'all_radars.shp'))
        radars = dict(zip(zip(df.lon, df.lat), df.radar.values))
    else:
        print('radar')
        season = kwargs.get('season', 'fall')
        radar_dir = osp.join(data_dir, data_source, season, year)
        print(radar_dir)
        radars = datahandling.load_radars(radar_dir)

    print('prepare static features')
    cells, radar_buffers, G, cell_to_radar_edges, radar_to_cell_edges = static_features(radars, **kwargs)

    # save to disk
    cells.to_file(osp.join(target_dir, 'tessellation.shp'))
    radar_buffers.to_file(osp.join(target_dir, 'radar_buffers.shp'))

    cell_df = cells.drop(columns='geometry')
    cell_df.to_csv(osp.join(target_dir, 'static_cell_features.csv'))

    radar_df = radar_buffers.drop(columns='geometry')
    radar_df.to_csv(osp.join(target_dir, 'static_radar_features.csv'))

    radar_to_cell_edges.to_csv(osp.join(target_dir, 'radar_to_cell_edges.csv'))
    cell_to_radar_edges.to_csv(osp.join(target_dir, 'cell_to_radar_edges.csv'))

    nx.write_graphml(G, osp.join(target_dir, 'delaunay.graphml'), infer_numeric_types=True)

    if kwargs.get('process_dynamic', True):
        # load dynamic features
        dynamic_feature_df, measurement_df = dynamic_features(data_dir, year, data_source,
                                                              cells, radar_buffers, **kwargs)
        # save to disk
        dynamic_feature_df.to_csv(osp.join(target_dir, 'dynamic_cell_features.csv'))
        measurement_df.to_csv(osp.join(target_dir, 'dynamic_radar_features.csv'))


def static_features(radars, **kwargs):
    """
    For a given set of radars, construct tessellation and associated static features.

    These features include: Delaunay triangulation (graph), cell areas, length and orientation of cell faces,
    distances between radars, and 25km buffers around radars used for bird estimation.

    :param radars: mapping from radar coordinates (lon, lat) to names
    :return: tessellation (geopandas dataframe), radar buffers (geopandas dataframe),
        Delaunay triangulation (networkx graph)
    """

    # check for radars to exclude
    exclude = []
    exclude.extend(kwargs.get('exclude', []))

    replacements = kwargs.get('replacements', {})

    for r1, r2 in replacements.items():
        # if two radars are available for the same location, use the first one
        if (r1 in radars.values()) and (r2 in radars.values()):
            exclude.append(r2)

    radars = {k: v for k, v in radars.items() if not v in exclude}
    print(f'radars: {radars}')
    print(f'excluded radars: {exclude}')


    # voronoi tesselation and associated graph
    edge_type = kwargs.get('edge_type', 'voronoi')
    ndummy = kwargs.get('n_dummy_radars', 0) if edge_type == 'voronoi' else 0
    space = spatial.Spatial(radars, n_dummy_radars=ndummy, buffer=kwargs.get('buffer', 150_000))

    # load nlcd land cover data as geopandas dataframe
    landcover_fp = kwargs.get('landcover_data', '')
    if osp.isfile(landcover_fp):
        landcover_gdf = gpd.read_file(landcover_fp)
    else:
        landcover_gdf = None

    if edge_type == 'hexagons':
        cells, G = space.hexagons(resolution=kwargs.get('h3_resolution'))
        cells = space.add_landcover_info(cells, landcover_gdf, on='h3_id')
    else:
        cells, G = space.voronoi()
        print(cells)
        print(cells.ID)
        print(landcover_gdf.ID)
        cells = space.add_landcover_info(cells, landcover_gdf, on='ID')

    print('create radar buffer dataframe')
    obs_range = kwargs.get('observation_range', 25_000)
    interp_range = kwargs.get('interpolation_range', 300_000)

    # 25 km buffers around radars
    radar_buffers = space.radar_buffers(obs_range)

    # compute areas of voronoi cells and radar buffers [unit is km^2]
    radar_buffers['area_km2'] = radar_buffers.area / 10**6
    cells['area_km2'] = cells.area / 10**6

    radar_buffers = radar_buffers.to_crs(space.crs_lonlat) #(f'epsg:{space.epsg_lonlat}')
    cells = cells.to_crs(space.crs_lonlat)

    # edges defining observation model from cells to radars
    cell_to_radar_edges = space.cell_to_radar_edges(obs_range)

    # edges defining interpolation model from radars to cells
    radar_to_cell_edges = space.radar_to_cell_edges(interp_range)

    print('done with static preprocessing')

    return cells, radar_buffers, G, cell_to_radar_edges, radar_to_cell_edges

def dynamic_features(data_dir, year, data_source, cells, radar_buffers, **kwargs):
    """
    Load all dynamic features, including bird densities and velocities, environmental data, and derived features
    such as estimated accumulation of bird on the ground due to adverse weather.

    Missing data is interpolated, but marked as missing.

    :param data_dir: directory containing all relevant data
    :param year: year of interest
    :param data_source: 'radar' or 'abm' (simulated data)
    :param voronoi: Voronoi tessellation (geopandas dataframe)
    :param radar_buffers: radar buffers with static features (geopandas dataframe)
    :return: dynamic features (pandas dataframe), measurements (pandas dataframe)
    """

    season = kwargs.get('season', 'fall')
    # pref_dirs = kwargs.get('pref_dirs', {'spring': 58, 'fall': 223})
    # pref_dir = pref_dirs[season]
    # wp_threshold = kwargs.get('wp_threshold', -0.5)
    edge_type = kwargs.get('edge_type', 'voronoi')
    t_unit = kwargs.get('t_unit', '1H')
    env_files = kwargs.get('env_files', ['model_levels.nc', 'surface.nc'])

    # load list of radars and time points to exclude due to rain and other artifacts
    excludes_path = osp.join(data_dir, data_source, kwargs.get('excludes', ''))
    if osp.isfile(excludes_path) and excludes_path.endswith('.csv'):
        print('load exclude file')
        df_excludes = pd.read_csv(excludes_path)
        df_excludes.start = pd.DatetimeIndex(df_excludes.start).tz_localize('UTC')
        df_excludes.end = pd.DatetimeIndex(df_excludes.end).tz_localize('UTC')
        df_excludes.radar = df_excludes.radar.str.lower()
    else:
        df_excludes = pd.DataFrame({'radar': [], 'start': [], 'end': []})
        print(f'did not find file {excludes_path}')
    print(df_excludes.head())

    print(f'##### load data for {season} {year} #####')

    if data_source in ['radar', 'nexrad']:
        print(f'load radar data')
        radar_dir = osp.join(data_dir, data_source)
        radar_names = radar_buffers.query('observed == True').radar.values

        radar_data, _, t_range = datahandling.load_season(radar_dir, season, year, ['vid', 'u', 'v'],
                                                          t_unit=t_unit, mask_days=False,
                                                          radar_names=radar_names,
                                                          interpolate_nans=False)

        birds_km2 = radar_data[:, 0, :]
        bird_u = radar_data[:, 1, :]
        bird_v = radar_data[:, 2, :]

        t_range = t_range.tz_localize('UTC')

    elif data_source == 'abm':
        print(f'load abm data')
        abm_dir = osp.join(data_dir, 'abm')
        observed_cells = cells.query('observed == True')

        data, t_range, bird_u, bird_v = abm.load_season(abm_dir, season, year, observed_cells)

        # rescale to birds per km^2
        birds_km2 = data / observed_cells.area_km2.to_numpy()[:, None]

    # time range for solar positions to be able to infer dusk and dawn
    solar_t_range = t_range.insert(-1, t_range[-1] + pd.Timedelta(t_range.freq))

    T = len(t_range)

    print('load env data')

    if len(env_files) > 0:
        if edge_type == 'none':
            env_areas = radar_buffers.geometry
        else:
            env_areas = cells.geometry
        # env_850 = era5interface.compute_cell_avg(osp.join(data_dir, 'env', data_source, season, year, 'pressure_level_850.nc'),
        #                                      env_areas, env_points,
        #                                      t_range.tz_localize(None), vars=env_vars, seed=random_seed)
        # env_surface = era5interface.compute_cell_avg(osp.join(data_dir, 'env', data_source, season, year, 'surface.nc'),
        #                                      env_areas, env_points,
        #                                      t_range.tz_localize(None), vars=env_vars, seed=random_seed)

        # env_850 = era5interface.extract_points(osp.join(data_dir, 'env', data_source, season, year, 'pressure_level_850.nc'),
        #                                      cells.lon.values, cells.lat.values, t_range.tz_localize(None), vars=env_vars)
        # env_surface = era5interface.extract_points(osp.join(data_dir, 'env', data_source, season, year, 'surface.nc'),
        #                                      cells.lon.values, cells.lat.values, t_range.tz_localize(None), vars=env_vars)

        # env_850 = era5interface.compute_cell_avg(
        #     osp.join(data_dir, 'env', data_source, season, year, 'pressure_level_850.nc'),
        #     env_areas, t_range.tz_localize(None), vars=env_vars)
        #
        # env_surface = era5interface.compute_cell_avg(osp.join(data_dir, 'env', data_source, season, year, 'surface.nc'),
        #                                            env_areas, t_range.tz_localize(None), vars=env_vars)
        # env_data = env_850.update(env_surface)

        # env_850_radars = era5interface.extract_points(
        #     osp.join(data_dir, 'env', data_source, season, year, 'pressure_level_850.nc'),
        #     radar_buffers.lon.values, radar_buffers.lat.values, t_range.tz_localize(None), vars=env_vars)
        # # env_surface_radars = era5interface.extract_points(osp.join(data_dir, 'env', data_source, season, year, 'surface.nc'),
        #                                            radar_buffers.lon.values, radar_buffers.lat.values, t_range.tz_localize(None),
        #                                            vars=env_vars)
        #
        # env_data_radars = env_850_radars.merge(env_surface_radars)
        print(t_range.tz_localize(None))
        env_data = {}
        for fname in env_files:
            env_data.update(
                era5interface.compute_cell_avg(
                    osp.join(data_dir, 'env', data_source, season, year, fname),
                    env_areas, t_range.tz_localize(None)
                )
            )


    dfs = []
    for ridx, row in cells.iterrows():

        df = {}

        df['ID'] = [row.ID] * len(t_range)

        # time related variables for radar ridx
        solarpos = np.array(solarposition.get_solarposition(solar_t_range, row.lat, row.lon).elevation)
        night = np.logical_or(solarpos[:-1] < -6, solarpos[1:] < -6)
        df['solarpos_dt'] = solarpos[:-1] - solarpos[1:]
        df['solarpos'] = solarpos[:-1]
        df['night'] = night
        df['dusk'] = np.logical_and(solarpos[:-1] >=6, solarpos[1:] < 6)  # switching from day to night
        df['dawn'] = np.logical_and(solarpos[:-1] < 6, solarpos[1:] >=6)  # switching from night to day
        df['datetime'] = t_range
        df['tidx'] = np.arange(t_range.size)


        if len(env_data) > 0:
            # environmental variables for cell ridx
            for var, values in env_data.items():
                df[var] = values[:, ridx]

            # compute accumulation variables (for baseline models)
            groups = [list(g) for k, g in it.groupby(enumerate(df['night']), key=lambda x: x[-1])]
            nights = [[item[0] for item in g] for g in groups if g[0][1]]
            df['nightID'] = np.zeros(t_range.size)
            df['frac_night_fw'] = np.zeros(t_range.size)
            df['frac_night_bw'] = np.zeros(t_range.size)
            #df['acc_rain'] = np.zeros(t_range.size)
            #df['acc_wind'] = np.zeros(t_range.size)
            #df['wind_profit'] = np.zeros(t_range.size)
            #acc_rain = 0
            #u_rain = 0
            #acc_wind = 0
            #u_wind = 0
            for nidx, night in enumerate(nights):
                df['nightID'][night] = np.ones(len(night)) * (nidx + 1)

                # relative night time
                df['frac_night_fw'][night] = np.arange(1, len(night) + 1) / len(night) # increasing
                df['frac_night_bw'][night] = np.arange(len(night), 0, -1) / len(night) # decreasing

                # accumulation due to rain in the past nights
                #acc_rain = acc_rain/3 + u_rain * 2/3
                #df['acc_rain'][night] = np.ones(len(night)) * acc_rain
                # compute proportion of hours with rain during the night
                #u_rain = np.mean(df['tp'][night] > 0.01)

                # accumulation due to unfavourable wind in the past nights
                #acc_wind = acc_wind/3 + u_wind * 2/3
                #df['acc_wind'][night] = np.ones(len(night)) * acc_wind
                # compute wind profit for bird with speed 12 m/s and flight direction 223 degree north
                #v_air = np.ones(len(night)) * 12
                #alpha = np.ones(len(night)) * pref_dir
                #df['wind_profit'][night] = v_air - np.sqrt(v_air**2 + df['wind_speed'][night]**2 -
                #                                           2 * v_air * df['wind_speed'][night] *
                #                                           np.cos(np.deg2rad(alpha-df['wind_dir'][night])))
                #u_wind = np.mean(df['wind_profit'][night]) < wp_threshold
        cell_df = pd.DataFrame(df)
        dfs.append(cell_df)


    dynamic_feature_df = pd.concat(dfs, ignore_index=True)
    print(f'feature columns: {dynamic_feature_df.columns}')

    dfs = []
    for ridx, row in radar_buffers.iterrows():

        df = {}

        df['radar'] = [row.radar] * len(t_range)
        df['ID'] = [row.ID] * len(t_range)

        # time related variables for radar ridx
        solarpos = np.array(solarposition.get_solarposition(solar_t_range, row.lat, row.lon).elevation)
        df['night'] = np.logical_or(solarpos[:-1] < -6, solarpos[1:] < -6)
        df['datetime'] = t_range
        df['tidx'] = np.arange(t_range.size)
        df['solarpos_dt'] = solarpos[:-1] - solarpos[1:]
        df['solarpos'] = solarpos[:-1]

        # bird related columns
        cols = ['birds_km2', 'bird_u', 'bird_v']

        # bird measurements for radar ridx
        if row.observed:
            df['birds_km2'] = birds_km2[ridx]
            df['bird_u'] = bird_u[ridx]
            df['bird_v'] = bird_v[ridx]
            df['missing_birds_km2'] = [0] * T
            df['missing_birds_uv'] = [0] * T
        else:
            df['birds_km2'] = [np.nan] * T
            df['bird_u'] = [np.nan] * T
            df['bird_v'] = [np.nan] * T
            df['missing_birds_km2'] = [1] * T
            df['missing_birds_uv'] = [1] * T

        radar_df = pd.DataFrame(df)

        # find time points to exclude for radars within this cell
        for edx, exclude in df_excludes.query(f'radar == "{row.radar}"').iterrows():
            radar_df['missing_birds_km2'] += ((t_range >= exclude.start) & (t_range <= exclude.end))
            radar_df['missing_birds_uv'] += ((t_range >= exclude.start) & (t_range <= exclude.end))
        radar_df['missing_birds_km2'] = radar_df['missing_birds_km2'].astype(bool)
        radar_df['missing_birds_uv'] = radar_df['missing_birds_uv'].astype(bool)

        # set bird quantities to NaN for these time points
        radar_df.loc[radar_df['missing_birds_km2'], 'birds_km2'] = np.nan
        radar_df.loc[radar_df['missing_birds_uv'], ['bird_u', 'bird_v']] = np.nan

        for col in cols:
            # radar quantities being exactly 0 during the night are missing,
            # radar quantities during the day are set to 0
            radar_df[col] = radar_df.apply(lambda row: np.nan if (row.night and not row[col])
            else (0 if not row.night else row[col]), axis=1)

            print(f'check missing data for column {col}')

            # remember missing radar observations
            if col == 'birds_km2':
                radar_df['missing_birds_km2'] = radar_df['missing_birds_km2'] | radar_df[col].isna()
            else:
                radar_df['missing_birds_uv'] = radar_df['missing_birds_uv'] | radar_df[col].isna()

            # interpolate linearly to fill missing data points
            radar_df[col] = radar_df[col].interpolate(method='linear', limit_direction='both')

        dfs.append(radar_df)
        print(f'found {radar_df.missing_birds_km2.sum()} missing birds_km2 time points')
        print(f'found {radar_df.missing_birds_uv.sum()} missing birds_uv time points')


    measurement_df = pd.concat(dfs, ignore_index=True)
    print(f'measurement columns: {measurement_df.columns}')

    return dynamic_feature_df, measurement_df
