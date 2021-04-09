import torch
from torch_geometric.data import Data, DataLoader, Dataset, InMemoryDataset
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


def static_features(data_dir, season, year):
    # load radar info
    radar_dir = osp.join(data_dir, 'radar', season, year)
    radars = datahandling.load_radars(radar_dir)

    # voronoi tesselation and associated graph
    space = spatial.Spatial(radars)
    voronoi, G = space.voronoi()
    G = space.subgraph('type', 'measured')  # graph without sink nodes

    # 25 km buffers around radars
    radar_buffers = gpd.GeoDataFrame({'radar': voronoi.radar},
                                     geometry=space.pts_local.buffer(25_000),
                                     crs=space.crs_local)

    # compute areas of voronoi cells and radar buffers [unit is km^2]
    radar_buffers['area_km2'] = radar_buffers.to_crs(epsg=space.epsg_equal_area).area / 10**6
    voronoi['area_km2'] = voronoi.to_crs(epsg=space.epsg_equal_area).area / 10**6

    return voronoi, radar_buffers, G

def dynamic_features(data_dir, data_source, season, year, voronoi, radar_buffers,
                     env_vars=['u', 'v', 'cc', 'tp', 'sp', 't2m', 'sshf'],
                     env_points=100, random_seed=1234, pref_dir=223, wp_threshold=-0.5):

    print(f'##### load data for {season} {year} #####')

    if data_source == 'radar':
        print(f'load radar data')
        radar_dir = osp.join(data_dir, 'radar')
        data, _, t_range = datahandling.load_season(radar_dir, season, year, 'vid',
                                                    mask_days=False, radar_names=voronoi.radar)
        data = data * voronoi.area_km2.to_numpy()[:, None] # rescale according to voronoi cell size
        t_range = t_range.tz_localize('UTC')

    elif data_source == 'abm':
        print(f'load abm data')
        abm_dir = osp.join(data_dir, 'abm')
        data, t_range = abm.load_season(abm_dir, season, year, voronoi)
        buffer_data, _ = abm.load_season(abm_dir, season, year, radar_buffers)
        buffer_data = buffer_data / radar_buffers.area_km2.to_numpy()[:, None] # rescale to birds per km^2
        buffer_data = buffer_data * voronoi.area_km2.to_numpy()[:, None] # rescale to birds per voronoi cell

    # time range for solar positions to be able to infer dusk and dawn
    solar_t_range = t_range.insert(-1, t_range[-1] + pd.Timedelta(t_range.freq))
    #solar_t_range = solar_t_range.insert(0, t_range[0] - pd.Timedelta(t_range.freq))

    print('load env data')
    env_850 = era5interface.compute_cell_avg(osp.join(data_dir, 'env', season, year, 'pressure_level_850.nc'),
                                         voronoi.geometry, env_points,
                                         t_range.tz_localize(None), vars=env_vars, seed=random_seed)
    env_surface = era5interface.compute_cell_avg(osp.join(data_dir, 'env', season, year, 'surface.nc'),
                                         voronoi.geometry, env_points,
                                         t_range.tz_localize(None), vars=env_vars, seed=random_seed)

    dfs = []
    for ridx, row in voronoi.iterrows():

        df = {}

        # bird measurementf for radar ridx
        df['birds'] = data[ridx]
        if data_source == 'abm':
            df['birds_from_buffer'] = buffer_data[ridx]
        else:
            df['birds_from_buffer'] = data[ridx]
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

        # environmental variables for radar ridx
        for var in env_vars:
            if var in env_850:
                df[var] = env_850[var][ridx]
            elif var in env_surface:
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
            u_rain = np.sum(np.where(df['tp'][night] > 0.01))

            # accumulation due to unfavourable wind in the past nights
            acc_wind = acc_wind/3 + u_wind * 2/3
            df['acc_wind'][night] = np.ones(len(night)) * acc_wind
            # compute wind profit for bird with speed 12 m/s and flight direction 223 degree north
            v_air = np.ones(len(night)) * 12
            alpha = np.ones(len(night)) * pref_dir
            df['wind_profit'][night] = v_air - np.sqrt(v_air**2 + df['wind_speed'][night]**2 - 2 * v_air *
                                                       df['wind_speed'] * np.cos(np.deg2rad(alpha-df['wind_dir'])))
            u_wind = np.mean(df['wind_profit'][night]) < wp_threshold


        dfs.append(pd.DataFrame(df))

    dynamic_feature_df = pd.concat(dfs, ignore_index=True)
    return dynamic_feature_df


def prepare_features(target_dir, data_dir, data_source, season, year, radar_years=['2015', '2016', '2017'],
                     env_vars=['u', 'v', 'cc', 'tp', 'sp', 't2m', 'sshf'],
                     env_points=100, random_seed=1234, pref_dirs={'spring': 58, 'fall': 223}, wp_threshold=-0.5):

    # load static features
    if data_source == 'abm' and not year in radar_years:
        radar_year = radar_years[-1]
    else:
        radar_year = year
    voronoi, radar_buffers, G = static_features(data_dir, season, radar_year)

    # save to disk
    voronoi.to_file(osp.join(target_dir, 'voronoi.shp'))
    radar_buffers.to_file(osp.join(target_dir, 'radar_buffers.shp'))
    nx.write_gpickle(G, osp.join(target_dir, 'delaunay.gpickle'), protocol=4)

    # load dynamic features
    dynamic_feature_df = dynamic_features(data_dir, data_source, season, year, voronoi, radar_buffers,
                                          env_vars=env_vars, env_points=env_points,
                                          random_seed=random_seed, pref_dir=pref_dirs[season],
                                          wp_threshold=wp_threshold)

    # save to disk
    dynamic_feature_df.to_csv(osp.join(target_dir, 'dynamic_features.csv'))


def angle(x1, y1, x2, y2):
    # for coords given in lonlat crs
    y = y1 - y2
    x = x1 - x2
    rad = np.arctan2(y, x)
    deg = np.rad2deg(rad)
    deg = (deg + 360) % 360
    return deg

def distance(x1, y1, x2, y2):
    # for coord1 and coord2 given in equidistant crs
    return np.linalg.norm(np.array([x1-x2, y1-y2])) / 10**3 # in kilometers


def rescale(features, min=None, max=None):
    if min is None:
        min = np.min(features)
    if max is None:
        max = np.max(features)
    if type(features) is not np.ndarray:
        features = np.array(features)
    return (features - min) / (max - min)

def reshape(data, nights, mask, timesteps):
    reshaped = [timeslice(data, night[0], mask, timesteps) for night in nights]
    reshaped = [d for d in reshaped if d.size > 0] # only use sequences that are fully available
    reshaped = np.stack(reshaped, axis=-1)
    return reshaped

def timeslice(data, start_night, mask, timesteps):
    data_night = data[..., start_night:]
    # remove hours during the day

    data_night = data_night[..., mask[start_night:]]
    if data_night.shape[-1] > timesteps:
        data_night = data_night[..., :timesteps+1]
    else:
        data_night = np.empty(0)
    return data_night

class Normalization:
    def __init__(self, root, years, season, data_source, radar_years=['2015', '2016', '2017'],
                 env_vars=['u', 'v', 'cc', 'tp', 'sp', 't2m', 'sshf'], env_points=100, seed=1234,
                 pref_dirs={'spring': 58, 'fall': 223}, wp_threshold=-0.5):
        self.root = root
        self.data_source = data_source
        self.season = season

        all_dfs = []
        for year in years:
            dir = self.preprocessed_dir(year)
            if not osp.isdir(dir):
                # load all features and organize them into dataframes
                os.makedirs(dir)
                prepare_features(dir, self.raw_dir, data_source, season, str(year),
                                 radar_years=radar_years, env_vars=env_vars,
                                 env_points=env_points, random_seed=seed,
                                 pref_dir=pref_dirs[season], wp_threshold=wp_threshold)

            # load features
            dynamic_feature_df = pd.read_csv(osp.join(self.preprocessed_dir(year), 'dynamic_features.csv'))
            all_dfs.append(dynamic_feature_df)
        self.feature_df = pd.concat(all_dfs)

    def normalize(self, data, key):
        min = self.min(key)
        max = self.max(key)
        data = (data - min) / (max - min)
        return data

    def min(self, key):
        return self.feature_df[key].dropna().min()

    def max(self, key):
        return self.feature_df[key].dropna().max()

    def preprocessed_dir(self, year):
        return osp.join(self.root, 'preprocessed', self.data_source, self.season, str(year))

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')


class RadarData(InMemoryDataset):

    def __init__(self, root, year, season, timesteps, transform=None, pre_transform=None, **kwargs):

        self.season = season
        self.year = str(year)
        self.timesteps = timesteps

        self.data_source = kwargs.get('data_source', 'radar')
        self.use_buffers = kwargs.get('use_buffers', False)
        self.bird_scale = kwargs.get('bird_scale', 1)
        self.env_points = kwargs.get('env_points', 100)
        self.radar_years = kwargs.get('radar_years', ['2015', '2016', '2017'])
        #self.env_vars = kwargs.get('env_vars', ['u', 'v'])
        self.env_vars = kwargs.get('surface_vars', ['u', 'v', 'cc', 'tp', 'sp', 't2m', 'sshf'])
        #self.surface_vars = kwargs.get('surface_vars', [])
        self.multinight = kwargs.get('multinight', True)
        self.random_seed = kwargs.get('seed', 1234)
        self.pref_dirs = kwargs.get('pref_dirs', {'spring': 58, 'fall': 223})
        self.wp_threshold = kwargs.get('wp_threshold', -0.5)

        self.start = kwargs.get('start', None)
        self.end = kwargs.get('end', None)
        self.normalize_dynamic = kwargs.get('normalize_dynamic', True)
        self.normalization = kwargs.get('normalization', None)


        if self.use_buffers:
            self.processed_dirname = f'measurements=from_buffers'
        else:
            self.processed_dirname = f'measurements=voronoi_cells'

        super(RadarData, self).__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

        with open(osp.join(self.processed_dir, self.info_file_name), 'rb') as f:
            self.info = pickle.load(f)

    @property
    def raw_file_names(self):
        return []

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')

    @property
    def preprocessed_dir(self):
        return osp.join(self.root, 'preprocessed', self.data_source, self.season, self.year)

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed', self.processed_dirname, self.data_source, self.season, self.year)

    @property
    def processed_file_names(self):
        return [f'data_timesteps={self.timesteps}.pt']

    @property
    def info_file_name(self):
        return f'info_timesteps={self.timesteps}.pkl'

    def download(self):
        pass


    def process(self):

        if not osp.isdir(self.preprocessed_dir):
            # load all features and organize them into dataframes
            os.makedirs(self.preprocessed_dir)
            prepare_features(self.preprocessed_dir, self.raw_dir, self.data_source, self.season, self.year,
                             radar_years=self.radar_years, env_vars=self.env_vars,
                             env_points=self.env_points, random_seed=self.random_seed, pref_dirs=self.pref_dirs,
                             wp_threshold=self.wp_threshold)

        # load features
        dynamic_feature_df = pd.read_csv(osp.join(self.preprocessed_dir, 'dynamic_features.csv'))
        voronoi = gpd.read_file(osp.join(self.preprocessed_dir, 'voronoi.shp'))
        G = nx.read_gpickle(osp.join(self.preprocessed_dir, 'delaunay.gpickle'))

        print('number of nans: ', dynamic_feature_df.birds.isna().sum())
        print('max bird measurement', dynamic_feature_df.birds.max())

        dynamic_feature_df.birds.fillna(0, inplace=True)

        # extract edges from graph
        edges = torch.tensor(list(G.edges()), dtype=torch.long)
        edge_index = edges.t().contiguous()

        # get distances, angles and face lengths between radars
        distances = rescale(np.array([data['distance'] for i, j, data in G.edges(data=True)]))
        angles = rescale(np.array([data['angle'] for i, j, data in G.edges(data=True)]), min=0, max=360)
        face_lengths = rescale(np.array([data['face_length'] for i, j, data in G.edges(data=True)]))


        # # normalize dynamic features
        # if self.normalize_dynamic:
        #     cidx = ~dynamic_feature_df.columns.isin(['birds', 'birds_from_buffer', 'radar', 'night',
        #                                              'dusk', 'dawn', 'datetime'])
        #     dynamic_feature_df.loc[:, cidx] = dynamic_feature_df.loc[:, cidx].apply(
        #         lambda col: (col - col.min()) / (col.max() - col.min()), axis=0)
        #     dynamic_feature_df['birds'] = dynamic_feature_df.birds / self.bird_scale
        #     if self.use_buffers:
        #         dynamic_feature_df['birds_from_buffer'] = dynamic_feature_df.birds_from_buffer / self.bird_scale
        if self.normalization is not None:
            cidx = ~dynamic_feature_df.columns.isin(['birds', 'birds_from_buffer', 'radar', 'night',
                                                     'dusk', 'dawn', 'datetime'])
            dynamic_feature_df.loc[:, cidx] = dynamic_feature_df.loc[:, cidx].apply(
                         lambda col: (col - self.normalization.min(col.name)) /
                                     (self.normalization.max(col.name) - self.normalization.min(col.name)), axis=0)
            self.bird_scale = self.normalization.max('birds')
            dynamic_feature_df['birds'] = dynamic_feature_df.birds / self.bird_scale
            if self.use_buffers:
                dynamic_feature_df['birds_from_buffer'] = dynamic_feature_df.birds_from_buffer / self.bird_scale



        input_col = 'birds_from_buffer' if self.use_buffers else 'birds'
        target_col = input_col
        self.suface_vars.remove('u')
        self.suface_vars.remove('v')
        env_cols = ['wind_speed', 'wind_dir', 'solarpos', 'solarpos_dt'] + self.surface_vars
        coord_cols = ['x', 'y']

        time = dynamic_feature_df.datetime.sort_values().unique()
        dayofyear = pd.DatetimeIndex(time).dayofyear.values
        tidx = np.arange(len(time))

        # normalize static features
        cidx = ['area_km2', *coord_cols]
        static = voronoi.loc[:, cidx].apply(lambda col: (col - col.min()) / (col.max() - col.min()), axis=0)
        areas = static.area_km2.to_numpy()
        coords = static[coord_cols].to_numpy()
        dayofyear = dayofyear / max(dayofyear)

        data = dict(inputs=[],
                    targets=[],
                    env=[],
                    nighttime=[],
                    dusk=[],
                    dawn=[])

        groups = dynamic_feature_df.groupby('radar')
        for name in voronoi.radar:
            df = groups.get_group(name).sort_values(by='datetime').reset_index(drop=True)
            data['inputs'].append(df[input_col].to_numpy())
            data['targets'].append(df[target_col].to_numpy())
            data['env'].append(df[env_cols].to_numpy().T)
            data['nighttime'].append(df.night.to_numpy())
            data['dusk'].append(df.dusk.to_numpy())
            data['dawn'].append(df.dawn.to_numpy())

        for k, v in data.items():
            data[k] = np.stack(v, axis=0)


        # find timesteps where it's night for all radars
        check_all = data['nighttime'].all(axis=0) # day/night mask
        # find timesteps where it's night for at least one radar
        check_any = data['nighttime'].any(axis=0)
        # also include timesteps immediately before dusk
        check_any = np.append(np.logical_or(check_any[:-1], check_any[1:]), check_any[-1])
        # dft = pd.DataFrame({'check_all': np.append(np.logical_and(check_all[:-1], check_all[1:]), False),
        #                     'check_any': np.append(np.logical_and(check_any[:-1], check_any[1:]), False),
        #                     'tidx': range(len(time))}, index=time)

        # group into nights
        groups = [list(g) for k, g in it.groupby(enumerate(check_all), key=lambda x: x[-1])]
        nights = [[item[0] for item in g] for g in groups if g[0][1]]

        # global_dusk_idx = [night[0] for night in nights]
        # global_dusk = np.zeros(tidx.shape)
        # global_dusk[global_dusk_idx] = 1

        if self.multinight:
            #mask = check_any
            mask = np.ones(check_any.shape, dtype=bool)
        else:
            mask = check_all

        for k, v in data.items():
            data[k] = reshape(v, nights, mask, self.timesteps)

        tidx = reshape(tidx, nights, mask, self.timesteps)
        dayofyear = reshape(dayofyear, nights, mask, self.timesteps)


        # set bird densities during the day to zero
        data['inputs'] = data['inputs'] * data['nighttime']
        data['targets'] = data['inputs'] * data['nighttime']

        edge_weights = np.exp(-np.square(distances) / np.square(np.std(distances)))
        R, T, N = data['inputs'].shape

        # create graph data objects per night
        data_list = [Data(x=torch.tensor(data['inputs'][:, :, nidx], dtype=torch.float),
                          y=torch.tensor(data['targets'][:, :, nidx], dtype=torch.float),
                          coords=torch.tensor(coords, dtype=torch.float),
                          areas=torch.tensor(areas, dtype=torch.float),
                          env=torch.tensor(data['env'][..., nidx], dtype=torch.float),
                          edge_index=edge_index,
                          edge_attr=torch.stack([
                              torch.tensor(distances, dtype=torch.float),
                              torch.tensor(angles, dtype=torch.float),
                              torch.tensor(face_lengths, dtype=torch.float)
                          ], dim=1),
                          edge_weight=torch.tensor(edge_weights, dtype=torch.float),
                          tidx=torch.tensor(tidx[:, nidx], dtype=torch.long),
                          day_of_year=torch.tensor(dayofyear[:, nidx], dtype=torch.float),
                          local_night=torch.tensor(data['nighttime'][:, :, nidx], dtype=torch.bool),
                          local_dusk=torch.tensor(data['dusk'][:, :, nidx], dtype=torch.bool),
                          local_dawn=torch.tensor(data['dawn'][:, :, nidx], dtype=torch.bool))
                     for nidx in range(N)]

        # write data to disk
        os.makedirs(self.processed_dir, exist_ok=True)

        info = {'radars': voronoi.radar.values,
                 'timepoints': time,
                 'tidx': tidx,
                 'nights': nights,
                 'bird_scale': self.bird_scale,
                 'boundaries': voronoi['boundary'].to_dict()}

        with open(osp.join(self.processed_dir, self.info_file_name), 'wb') as f:
            pickle.dump(info, f)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


