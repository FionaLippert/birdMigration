import torch
from torch_geometric.data import Data, DataLoader, Dataset, InMemoryDataset
import numpy as np
import networkx as nx
import os.path as osp
import os
import pandas as pd
import pickle5 as pickle
import itertools as it


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
        min = np.nanmin(features)
    if max is None:
        max = np.nanmax(features)
    if type(features) is not np.ndarray:
        features = np.array(features)

    rescaled = features - min
    if max != min:
        rescaled /= (max - min)
    return rescaled

def reshape(data, nights, mask, timesteps, use_nights=True, index=None):
    if use_nights:
        reshaped = reshape_nights(data, nights, mask, timesteps)
    else:
        reshaped = reshape_t(data, timesteps, index)
    return reshaped

def reshape_nights(data, nights, mask, timesteps):
    reshaped = [timeslice(data, night[0], mask, timesteps) for night in nights]
    reshaped = [d for d in reshaped if d.size > 0] # only use sequences that are fully available
    reshaped = np.stack(reshaped, axis=-1)
    return reshaped

def reshape_t(data, timesteps, index=None):

    # reshaped = [data[..., t:t+timesteps+1] for t in np.arange(0, data.shape[-1] - timesteps - 1)]
    if index is None:
        index = np.arange(0, data.shape[-1] - timesteps - 1)
    reshaped = [data[..., t:t + timesteps + 1] for t in index]
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
    def __init__(self, years, data_source, data_root, season='fall', radar_years=['2015', '2016', '2017'], max_distance=216,
                 env_points=100, seed=1234, pref_dirs={'spring': 58, 'fall': 223}, wp_threshold=-0.5, t_unit='1H',
                 edge_type='voronoi', n_dummy_radars=0, exclude=[], **kwargs):
        self.root = data_root
        self.data_source = data_source
        self.season = season
        self.t_unit = t_unit
        self.edge_type = edge_type
        self.n_dummy_radars = n_dummy_radars
        self.exclude = exclude

        all_dfs = []
        for year in years:
            print('load year', year)
            dir = self.preprocessed_dir(year)
            if not osp.isdir(dir):
                # preprocessed data is not available
                print('Preprocessed data not available. Please run preprocessing script first.')

            # load features
            dynamic_feature_df = pd.read_csv(osp.join(self.preprocessed_dir(year), 'dynamic_features.csv'))
            all_dfs.append(dynamic_feature_df)
        self.feature_df = pd.concat(all_dfs)

    def normalize(self, data, key):
        min = self.min(key)
        max = self.max(key)
        data = (data - min) / (max - min)
        return data

    def denormalize(self, data, key):
        min = self.min(key)
        max = self.max(key)
        data = data * (max - min) + min
        return data

    def min(self, key):
        return self.feature_df[key].dropna().min()

    def max(self, key):
        return self.feature_df[key].dropna().max()

    def root_min(self, key, root):
        root_transformed = self.feature_df[key].apply(lambda x: np.power(x, 1/root))
        return root_transformed.dropna().min()

    def root_max(self, key, root):
        root_transformed = self.feature_df[key].apply(lambda x: np.power(x, 1/root))
        return root_transformed.dropna().max()

    def preprocessed_dir(self, year):
        return osp.join(self.root, 'preprocessed', self.t_unit, f'{self.edge_type}_dummy_radars={self.n_dummy_radars}_exclude={self.exclude}',
                        self.data_source, self.season, str(year))

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')

def angle(coord1, coord2):
    # coords should be in lonlat crs
    y = coord2[0] - coord1[0]
    x = coord2[1] - coord1[1]

    rad = np.arctan2(y, x)
    deg = np.rad2deg(rad)
    deg = (deg + 360) % 360

    return deg

def compute_flux(dens, ff, dd, alpha, l=1):
    # compute number of birds crossing transect of length 'l' [km] and angle 'alpha' per hour
    mtr = dens * ff * np.cos(np.deg2rad(dd - alpha))
    flux = mtr * l * 3.6
    return flux


class RadarData(InMemoryDataset):

    def __init__(self, year, timesteps, transform=None, pre_transform=None, **kwargs):

        self.root = kwargs.get('data_root')
        self.sub_dir = osp.join(self.root, kwargs.get('sub_dir', ''))
        self.season = kwargs.get('season')
        self.year = str(year)
        self.timesteps = timesteps

        self.data_source = kwargs.get('data_source', 'radar')
        self.use_buffers = kwargs.get('use_buffers', False)
        self.bird_scale = kwargs.get('bird_scale', 1)
        self.env_points = kwargs.get('env_points', 100)
        self.radar_years = kwargs.get('radar_years', ['2015', '2016', '2017'])
        #self.env_vars = kwargs.get('env_vars', ['u', 'v'])
        self.env_vars = kwargs.get('env_vars', ['u', 'v', 'cc', 'tp', 'sp', 't2m', 'sshf'])
        #self.surface_vars = kwargs.get('surface_vars', [])
        self.multinight = kwargs.get('multinight', True)
        self.random_seed = kwargs.get('seed', 1234)
        self.pref_dirs = kwargs.get('pref_dirs', {'spring': 58, 'fall': 223})
        self.wp_threshold = kwargs.get('wp_threshold', -0.5)
        self.root_transform = kwargs.get('root_transform', 0)
        self.missing_data_threshold = kwargs.get('missing_data_threshold', 0)

        self.start = kwargs.get('start', None)
        self.end = kwargs.get('end', None)
        self.normalize_dynamic = kwargs.get('normalize_dynamic', True)
        self.normalization = kwargs.get('normalization', None)

        self.edge_type = kwargs.get('edge_type', 'voronoi')
        self.max_distance = kwargs.get('max_distance', 216)
        self.t_unit = kwargs.get('t_unit', '1H')
        self.n_dummy_radars = kwargs.get('n_dummy_radars', 0)

        self.birds_per_km2 = kwargs.get('birds_per_km2', False)

        self.exclude = kwargs.get('exclude', [])

        self.compute_fluxes = kwargs.get('compute_fluxes', False)

        self.use_nights = kwargs.get('use_nights', True)
        self.seed = kwargs.get('seed', 1234)
        self.rng = np.random.default_rng(self.seed)
        self.data_perc = kwargs.get('data_perc', 1.0)

        measurements = 'from_buffers' if self.use_buffers else 'voronoi_cells'
        self.processed_dirname = f'measurements={measurements}_root_transform={self.root_transform}_use_nights={self.use_nights}_' \
                                 f'edges={self.edge_type}_birds_km2={self.birds_per_km2}_dummy_radars={self.n_dummy_radars}_t_unit={self.t_unit}_exclude={self.exclude}'

        super(RadarData, self).__init__(self.root, transform, pre_transform)

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
        return osp.join(self.root, 'preprocessed', self.t_unit, f'{self.edge_type}_dummy_radars={self.n_dummy_radars}_exclude={self.exclude}',
                        self.data_source, self.season, self.year)

    @property
    def processed_dir(self):
        return osp.join(self.sub_dir, 'processed', self.processed_dirname, self.data_source, self.season, self.year)

    @property
    def processed_file_names(self):
        return [f'data_timesteps={self.timesteps}.pt']

    @property
    def info_file_name(self):
        return f'info_timesteps={self.timesteps}.pkl'

    def download(self):
        pass


    def process(self):
        print(self.preprocessed_dir)
        if not osp.isdir(self.preprocessed_dir):
            print('Preprocessed data not available. Please run preprocessing script first.')
            # load all features and organize them into dataframes
            # os.makedirs(self.preprocessed_dir)
            # prepare_features(self.preprocessed_dir, self.raw_dir, self.data_source, self.season, self.year,
            #                  radar_years=self.radar_years,
            #                  env_points=self.env_points, random_seed=self.random_seed, pref_dirs=self.pref_dirs,
            #                  wp_threshold=self.wp_threshold, max_distance=self.max_distance, t_unit=self.t_unit,
            #                  edge_type=self.edge_type, n_dummy_radars=self.n_dummy_radars, exclude=self.exclude)

        # load features
        dynamic_feature_df = pd.read_csv(osp.join(self.preprocessed_dir, 'dynamic_features.csv'))
        voronoi = pd.read_csv(osp.join(self.preprocessed_dir, 'static_features.csv'))

        if self.edge_type == 'voronoi':
            G = nx.read_gpickle(osp.join(self.preprocessed_dir, 'delaunay.gpickle'))
        else:
            # print(f'create graph with max distance = {self.max_distance}')
            G_path = osp.join(self.preprocessed_dir, f'G_max_dist={self.max_distance}.gpickle')
            G = nx.read_gpickle(G_path)
            # if not osp.isfile(G_path):
            #     prepare_features(self.preprocessed_dir, self.raw_dir, self.data_source, self.season, self.year,
            #                      radar_years=self.radar_years, max_distance=self.max_distance, process_dynamic=False,
            #                      exclude=self.exclude)



        print('number of nans: ', dynamic_feature_df.birds.isna().sum())
        print('max bird measurement', dynamic_feature_df.birds.max())


        # extract edges from graph
        edges = torch.tensor(list(G.edges()), dtype=torch.long)
        edge_index = edges.t().contiguous()
        n_edges = edge_index.size(1)

        # boundary radars and boundary edges
        boundary = voronoi['boundary'].to_numpy()
        boundary2inner_edges = torch.tensor([(boundary[edge_index[0, idx]] and not boundary[edge_index[1, idx]])
                                            for idx in range(n_edges)])
        inner2boundary_edges = torch.tensor([(not boundary[edge_index[0, idx]] and boundary[edge_index[1, idx]])
                                             for idx in range(n_edges)])
        inner_edges = torch.tensor([(not boundary[edge_index[0, idx]] and not boundary[edge_index[1, idx]])
                                    for idx in range(n_edges)])
        boundary2boundary_edges = torch.tensor([(boundary[edge_index[0, idx]] and boundary[edge_index[1, idx]])
                                    for idx in range(n_edges)])

        reverse_edges = torch.zeros(n_edges, dtype=torch.long)
        for idx in range(n_edges):
            for jdx in range(n_edges):
                if (edge_index[:, idx] == torch.flip(edge_index[:, jdx], dims=[0])).all():
                    reverse_edges[idx] = jdx


        # get distances, angles and face lengths between radars
        # print(G.edges(data=True))
        # print('normalize distances')
        distances = rescale(np.array([data['distance'] for i, j, data in G.edges(data=True)]))
        # print('normalize angles')
        angles = rescale(np.array([data['angle'] for i, j, data in G.edges(data=True)]), min=0, max=360)


        if self.edge_type == 'voronoi':
            print('Use Voronoi tessellation')
            face_lengths = rescale(np.array([data['face_length'] for i, j, data in G.edges(data=True)]))
            edge_attr = torch.stack([
                                  torch.tensor(distances, dtype=torch.float),
                                  torch.tensor(angles, dtype=torch.float),
                                  torch.tensor(face_lengths, dtype=torch.float)
                              ], dim=1)
        else:
            print('Use other edge type')
            edge_attr = torch.stack([
                torch.tensor(distances, dtype=torch.float),
                torch.tensor(angles, dtype=torch.float),
            ], dim=1)


        # # normalize dynamic features
        # if self.normalize_dynamic:
        #     cidx = ~dynamic_feature_df.columns.isin(['birds', 'birds_from_buffer', 'radar', 'night',
        #                                              'dusk', 'dawn', 'datetime'])
        #     dynamic_feature_df.loc[:, cidx] = dynamic_feature_df.loc[:, cidx].apply(
        #         lambda col: (col - col.min()) / (col.max() - col.min()), axis=0)
        #     dynamic_feature_df['birds'] = dynamic_feature_df.birds / self.bird_scale
        #     if self.use_buffers:
        #         dynamic_feature_df['birds_from_buffer'] = dynamic_feature_df.birds_from_buffer / self.bird_scale

        if self.edge_type == 'voronoi' and not self.birds_per_km2:
            if self.use_buffers:
                input_col = 'birds_from_buffer'
            else:
                input_col = 'birds'
        else:
            input_col = 'birds_km2'

        print('input col', input_col)

        dynamic_feature_df['missing'] = dynamic_feature_df[input_col].isna() # remember which data was missing
        print(len(dynamic_feature_df), dynamic_feature_df[input_col].isna().sum())
        dynamic_feature_df[input_col].fillna(0, inplace=True)


        # apply root transform
        if self.root_transform > 0:
            dynamic_feature_df[input_col] = dynamic_feature_df[input_col].apply(
                                            lambda x: np.power(x, 1/self.root_transform))

        if self.normalization is not None:
            cidx = ~dynamic_feature_df.columns.isin([input_col, 'birds_km2', 'bird_speed', 'bird_direction',
                                                     'radar', 'night', 'boundary',
                                                     'dusk', 'dawn', 'datetime', 'missing'])
            dynamic_feature_df.loc[:, cidx] = dynamic_feature_df.loc[:, cidx].apply(
                         lambda col: (col - self.normalization.min(col.name)) /
                                     (self.normalization.max(col.name) - self.normalization.min(col.name)), axis=0)

            if self.root_transform > 0:
                self.bird_scale = self.normalization.root_max(input_col, self.root_transform)
            else:
                self.bird_scale = self.normalization.max(input_col)
            print(input_col)
            print(f'bird scale = {self.bird_scale}')
            # dynamic_feature_df['birds'] = dynamic_feature_df.birds / self.bird_scale
            # dynamic_feature_df['birds_from_buffer'] = dynamic_feature_df.birds_from_buffer / self.bird_scale
            # dynamic_feature_df['birds_km2'] = dynamic_feature_df.birds_km2 / self.bird_scale

            dynamic_feature_df[input_col] = dynamic_feature_df[input_col] / self.bird_scale
            if input_col != 'birds_km2':
                dynamic_feature_df['birds_km2'] = dynamic_feature_df['birds_km2'] / self.bird_scale
            #print('number of nans: ', dynamic_feature_df.birds_from_buffer.isna().sum())



        target_col = input_col
        # self.env_vars.remove('u')
        # self.env_vars.remove('v')

        # env_cols = ['wind_speed', 'wind_dir', 'solarpos', 'solarpos_dt'] + \
        #            [var for var in self.env_vars if not var in ['u', 'v']]
        env_cols =  [var for var in self.env_vars] + ['solarpos', 'solarpos_dt']
        acc_cols = ['acc_rain', 'acc_wind']
        coord_cols = ['x', 'y']
        # coord_cols = ['lon', 'lat']

        time = dynamic_feature_df.datetime.sort_values().unique()
        dayofyear = pd.DatetimeIndex(time).dayofyear.values
        tidx = np.arange(len(time))

        # normalize static features
        #cidx = ['area_km2']#, *coord_cols]
        areas = voronoi[['area_km2']].apply(lambda col: (col - col.min()) / (col.max() - col.min()), axis=0).to_numpy()

        if self.edge_type != 'voronoi':
            areas = np.ones(areas.shape)

        # coords = voronoi[coord_cols].apply(lambda col: np.radians(col)).to_numpy()
        coords = voronoi[coord_cols].apply(lambda col: (col - col.min()) / (col.max() - col.min()), axis=0).to_numpy()

        dayofyear = dayofyear / max(dayofyear)

        data = dict(inputs=[],
                    targets=[],
                    env=[],
                    acc=[],
                    nighttime=[],
                    dusk=[],
                    dawn=[],
                    missing=[])

        if self.data_source == 'radar' and self.compute_fluxes:
            data['speed'] = []
            data['direction'] = []
            data['birds_km2'] = []
            data['bird_uv'] = []

        groups = dynamic_feature_df.groupby('radar')
        for name in voronoi.radar:
            df = groups.get_group(name).sort_values(by='datetime').reset_index(drop=True)
            data['inputs'].append(df[input_col].to_numpy())
            data['targets'].append(df[target_col].to_numpy())
            data['env'].append(df[env_cols].to_numpy().T)
            data['acc'].append(df[acc_cols].to_numpy().T)
            data['nighttime'].append(df.night.to_numpy())
            data['dusk'].append(df.dusk.to_numpy())
            data['dawn'].append(df.dawn.to_numpy())
            data['missing'].append(df.missing.to_numpy())

            if self.data_source == 'radar' and self.compute_fluxes:
                data['speed'].append(df.bird_speed.to_numpy())
                data['direction'].append(df.bird_direction.to_numpy())
                data['birds_km2'].append(df.birds_km2.to_numpy())
                data['bird_uv'].append(df[['bird_u', 'bird_v']].to_numpy().T)

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

        if self.use_nights:
            seq_index = None
        else:
            n_seq = int(self.data_perc * (mask.shape[-1] - self.timesteps - 1))
            print(f'data_perc = {self.data_perc}')
            print(f'n_seq = {n_seq}')
            seq_index = self.rng.permutation(mask.shape[-1] - self.timesteps - 1)[:n_seq]

        for k, v in data.items():
            data[k] = reshape(v, nights, mask, self.timesteps, self.use_nights, seq_index)



        if self.data_source == 'radar' and self.compute_fluxes:
            print('compute fluxes')
            fluxes = []
            mtr = []
            for i, j, e_data in G.edges(data=True):
                vid_i = data['birds_km2'][i]
                vid_j = data['birds_km2'][j]
                vid_i[np.isnan(vid_i)] = vid_j[np.isnan(vid_i)]
                vid_j[np.isnan(vid_j)] = vid_i[np.isnan(vid_j)]

                dd_i = data['direction'][i]
                dd_j = data['direction'][j]
                dd_i[np.isnan(dd_i)] = dd_j[np.isnan(dd_i)]
                dd_j[np.isnan(dd_j)] = dd_i[np.isnan(dd_j)]

                ff_i = data['speed'][i]
                ff_j = data['speed'][j]
                ff_i[np.isnan(ff_i)] = ff_j[np.isnan(ff_i)]
                ff_j[np.isnan(ff_j)] = ff_i[np.isnan(ff_j)]

                vid_interp = (vid_i + vid_j) / 2
                dd_interp = ((dd_i + 360) % 360 + (dd_j + 360) % 360) / 2
                ff_interp = (ff_i + ff_j) / 2
                length = e_data.get('face_length', 1)
                fluxes.append(compute_flux(vid_interp, ff_interp, dd_interp, e_data['angle'], length))
                mtr.append(compute_flux(vid_interp, ff_interp, dd_interp, e_data['angle'], 1))
            fluxes = np.stack(fluxes, axis=0)
            mtr = np.stack(mtr, axis=0)
        else:
            fluxes = np.zeros((len(G.edges()), data['inputs'].shape[1], data['inputs'].shape[2]))
            mtr = np.zeros((len(G.edges()), data['inputs'].shape[1], data['inputs'].shape[2]))

            data['direction'] = np.zeros((len(G.nodes()), data['inputs'].shape[1], data['inputs'].shape[2]))
            data['speed'] = np.zeros((len(G.nodes()), data['inputs'].shape[1], data['inputs'].shape[2]))
            data['bird_uv'] = np.zeros((len(G.nodes()), data['inputs'].shape[1], data['inputs'].shape[2]))

        data['direction'] = (data['direction'] + 360) % 360
        data['direction'] = rescale(data['direction'], min=0, max=360)

        data['direction'][np.isnan(data['direction'])] = -1
        data['speed'] = (data['speed'] - self.normalization.min('bird_speed')) / (self.normalization.max('bird_speed')
                                                                                  - self.normalization.min('bird_speed'))
        data['speed'][np.isnan(data['speed'])] = -1
        data['bird_uv'][np.isnan(data['bird_uv'])] = 0 #TODO necessary?


        tidx = reshape(tidx, nights, mask, self.timesteps, self.use_nights)
        dayofyear = reshape(dayofyear, nights, mask, self.timesteps, self.use_nights)


        # set bird densities during the day to zero
        data['inputs'] = data['inputs'] * data['nighttime']
        data['targets'] = data['targets'] * data['nighttime']

        edge_weights = np.exp(-np.square(distances) / np.square(np.std(distances)))
        R, T, N = data['inputs'].shape

        # create graph data objects per night
        data_list = [Data(x=torch.tensor(data['inputs'][:, :, nidx], dtype=torch.float),
                          y=torch.tensor(data['targets'][:, :, nidx], dtype=torch.float),
                          coords=torch.tensor(coords, dtype=torch.float),
                          areas=torch.tensor(areas, dtype=torch.float),
                          boundary=torch.tensor(boundary, dtype=torch.bool),
                          env=torch.tensor(data['env'][..., nidx], dtype=torch.float),
                          acc=torch.tensor(data['acc'][..., nidx], dtype=torch.float),
                          edge_index=edge_index,
                          reverse_edges=reverse_edges,
                          boundary2inner_edges=boundary2inner_edges.bool(),
                          inner2boundary_edges=inner2boundary_edges.bool(),
                          boundary2boundary_edges=boundary2boundary_edges.bool(),
                          inner_edges=inner_edges.bool(),
                          edge_attr=edge_attr,
                          edge_weight=torch.tensor(edge_weights, dtype=torch.float),
                          tidx=torch.tensor(tidx[:, nidx], dtype=torch.long),
                          day_of_year=torch.tensor(dayofyear[:, nidx], dtype=torch.float),
                          local_night=torch.tensor(data['nighttime'][:, :, nidx], dtype=torch.bool),
                          local_dusk=torch.tensor(data['dusk'][:, :, nidx], dtype=torch.bool),
                          local_dawn=torch.tensor(data['dawn'][:, :, nidx], dtype=torch.bool),
                          missing=torch.tensor(data['missing'][:, :, nidx], dtype=torch.bool),
                          fluxes=torch.tensor(fluxes[:, :, nidx], dtype=torch.float),
                          mtr=torch.tensor(mtr[:, :, nidx], dtype=torch.float),
                          directions=torch.tensor(data['direction'][:, :, nidx], dtype=torch.float),
                          speeds=torch.tensor(data['speed'][:, :, nidx], dtype=torch.float),
                          bird_uv=torch.tensor(data['bird_uv'][..., nidx], dtype=torch.float))
                     for nidx in range(N) if data['missing'][:, :, nidx].mean() <= self.missing_data_threshold]

        print(f'number of sequences = {len(data_list)}')

        # write data to disk
        os.makedirs(self.processed_dir, exist_ok=True)
        n_seq_discarded = np.sum(data['missing'].mean((0, 1)) > self.missing_data_threshold)
        print(f'discarded {n_seq_discarded} sequences due to missing data')
        info = {'radars': voronoi.radar.values,
                'areas' : voronoi.area_km2.values,
                 'timepoints': time,
                 'tidx': tidx,
                 'nights': nights,
                 'bird_scale': self.bird_scale,
                 'boundaries': voronoi['boundary'].to_dict(),
                 'root_transform': self.root_transform,
                 'n_seq_discarded': n_seq_discarded}

        with open(osp.join(self.processed_dir, self.info_file_name), 'wb') as f:
            pickle.dump(info, f)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
