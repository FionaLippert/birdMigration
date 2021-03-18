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
    print(radar_dir)
    radars = datahandling.load_radars(radar_dir)
    print(radars)

    # voronoi tesselation and associated graph
    space = spatial.Spatial(radars)
    voronoi, G = space.voronoi()
    G = nx.DiGraph(space.subgraph('type', 'measured'))  # graph without sink nodes

    # 25 km buffers around radars
    radar_buffers = gpd.GeoDataFrame({'radar': voronoi.radar},
                                     geometry=space.pts_local.buffer(25_000),
                                     crs=f'EPSG:{space.epsg_equidistant}')

    # compute areas of voronoi cells and radar buffers [unit is km^2]
    radar_buffers['area_km2'] = radar_buffers.to_crs(epsg=space.epsg_equal_area).area / 10**6
    voronoi['area_km2'] = voronoi.to_crs(epsg=space.epsg_equal_area).area / 10**6

    return voronoi, radar_buffers, G

def dynamic_features(data_dir, data_source, season, year, voronoi, radar_buffers,
                     env_vars=['u', 'v'], env_points=100, random_seed=1234):

    print(f'##### load data for {season} {year} #####')

    if data_source == 'radar':
        print(f'load radar data')
        radar_dir = osp.join(data_dir, 'rad ar')
        data, _, t_range = datahandling.load_season(radar_dir, season, year, 'vid',
                                                    mask_days=False, radar_names=voronoi.radar)
        data = data * voronoi.area_km2.to_numpy()[:, None, None] # rescale according to voronoi cell size
        t_range = t_range.tz_localize('UTC')

    elif data_source == 'abm':
        print(f'load abm data')
        abm_dir = osp.join(data_dir, 'abm')
        data, t_range = abm.load_season(abm_dir, season, year, voronoi)
        buffer_data, _ = abm.load_season(abm_dir, season, year, radar_buffers)
        buffer_data = buffer_data / radar_buffers.area_km2.to_numpy()[:, None] # rescale to birds per km^2
        buffer_data = buffer_data * voronoi.area_km2.to_numpy()[:, None] # rescale to birds per voronoi cell

    print('load sun data')
    solar_t_range = t_range.insert(-1, t_range[-1] + pd.Timedelta(t_range.freq))

    print('load env data')
    env = era5interface.compute_cell_avg(osp.join(data_dir, 'env', season, year, 'pressure_level_850.nc'),
                                         voronoi.geometry, env_points,
                                         t_range.tz_localize(None), vars=env_vars, seed=random_seed)

    dfs = []
    for ridx, row in voronoi.iterrows():

        df = {}

        # bird measurementf for radar ridx
        df['birds'] = data[ridx]
        if data_source == 'abm':
            df['birds_from_buffer'] = buffer_data[ridx]
        df['radar'] = [row.radar] * len(t_range)

        # time related variables for radar ridx
        solarpos = np.array(solarposition.get_solarposition(solar_t_range, row.lat, row.lon).elevation)
        night = solarpos < -6
        df['solarpos_dt'] = solarpos[:-1] - solarpos[1:]
        df['solarpos'] = solarpos[:-1]
        df['night'] = night[:-1]
        df['dusk'] = np.logical_and(~night[:-1], night[1:])  # switching from day to night
        df['dawn'] = np.logical_and(night[:-1], ~night[1:])  # switching from night to day
        df['datetime'] = t_range

        # environmental variables for radar ridx
        for var in env_vars:
            df[var] = env[var][ridx]
        df['wind_speed'] = np.sqrt(np.square(df['u']) + np.square(df['v']))
        # Note that here wind direction is the direction into which the wind is blowing,
        # which is the opposite of the standard meteorological wind direction
        df['wind_dir'] = (abm.uv2deg(df['u'], df['v']) + 360) % 360

        dfs.append(pd.DataFrame(df))

    dynamic_feature_df = pd.concat(dfs, ignore_index=True)
    return dynamic_feature_df


def prepare_features(target_dir, data_dir, data_source, season, year,
                     radar_years=['2015', '2016', '2017'], env_vars=['u', 'v'],
                     env_points=100, random_seed=1234):

    # load static features
    if data_source == 'abm' and not year in radar_years:
        radar_year = radar_years[-1]
    else:
        radar_year = year
    print(radar_year)
    voronoi, radar_buffers, G = static_features(data_dir, season, radar_year)

    # save to disk
    voronoi.to_file(osp.join(target_dir, 'voronoi.shp'))
    radar_buffers.to_file(osp.join(target_dir, 'radar_buffers.shp'))
    nx.write_gpickle(G, osp.join(target_dir, 'delaunay.gpickle'), protocol=4)

    # load dynamic features
    dynamic_feature_df = dynamic_features(data_dir, data_source, season, year, voronoi, radar_buffers,
                                          env_vars, env_points, random_seed)

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


def normalize(features, min=None, max=None):
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


class RadarData(InMemoryDataset):

    def __init__(self, root, split, year, season='fall', timesteps=1,
                 data_source='radar', use_buffers=False, bird_scale = 2000, env_points=100,
                 radar_years=['2015', '2016', '2017'], env_vars=['u', 'v'], multinight=False,
                 start=None, end=None, transform=None, pre_transform=None):

        self.split = split
        self.season = season
        self.year = year
        self.timesteps = timesteps
        self.data_source = data_source
        self.start = start
        self.end = end
        self.bird_scale = bird_scale
        self.env_points = env_points # number of environment variable samples per radar cell
        self.radar_years = radar_years # years for which radar data is available
        self.env_vars = env_vars
        self.multinight = multinight
        self.use_buffers = use_buffers and data_source == 'abm'
        self.random_seed = 1234

        if self.use_buffers:
            self.processed_dirname = f'measurements=from_buffers_split={split}'
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
                             env_points=self.env_points, random_seed=self.random_seed)

        # load features
        dynamic_feature_df = pd.read_csv(osp.join(self.preprocessed_dir, 'dynamic_features.csv'))
        voronoi = gpd.read_file(osp.join(self.preprocessed_dir, 'voronoi.shp'))
        G = nx.read_gpickle(osp.join(self.preprocessed_dir, 'delaunay.gpickle'))

        # extract edges from graph
        edges = torch.tensor(list(G.edges()), dtype=torch.long)
        edge_index = edges.t().contiguous()

        # compute distances and angles between radars
        distances = normalize([distance(voronoi.x.iloc[j], voronoi.y.iloc[j],
                                        voronoi.x.iloc[i], voronoi.y.iloc[i]) for j, i in G.edges], min=0)
        angles = normalize([angle(voronoi.lon.iloc[j], voronoi.lat.iloc[j],
                                  voronoi.lon.iloc[i], voronoi.lat.iloc[i]) for j, i in G.edges], min=0, max=360)

        # normalize dynamic features
        cidx = ~dynamic_feature_df.columns.isin(['birds', 'birds_from_buffer', 'radar', 'night',
                                                 'dusk', 'dawn', 'datetime'])

        dynamic_feature_df.loc[:, cidx] = dynamic_feature_df.loc[:, cidx].apply(
            lambda col: (col - col.min()) / (col.max() - col.min()), axis=0)
        dynamic_feature_df['birds'] = dynamic_feature_df.birds / self.bird_scale
        if self.use_buffers:
            dynamic_feature_df['birds_from_buffer'] = dynamic_feature_df.birds_from_buffer / self.bird_scale


        input_col = 'birds_from_buffer' if self.use_buffers else 'birds'
        target_col = 'birds_from_buffer' if self.use_buffers and self.split == 'train' else 'birds'
        env_cols = ['wind_speed', 'wind_dir', 'solarpos', 'solarpos_dt']
        coord_cols = ['x', 'y']

        time = dynamic_feature_df.datetime.sort_values().unique()
        tidx = np.arange(len(time))

        # normalize static features
        cidx = ['area_km2', *coord_cols]
        static = voronoi.loc[:, cidx].apply(lambda col: (col - col.min()) / (col.max() - col.min()), axis=0)
        areas = static.area_km2.to_numpy()
        coords = static[coord_cols].to_numpy()

        inputs = []
        targets = []
        env = []
        night = []
        dusk = []
        dawn = []

        groups = dynamic_feature_df.groupby('radar')
        for name in voronoi.radar:
            df = groups.get_group(name).sort_values(by='datetime').reset_index(drop=True)
            inputs.append(df[input_col].to_numpy())
            targets.append(df[target_col].to_numpy())
            env.append(df[env_cols].to_numpy().T)
            night.append(df.night.to_numpy())
            dusk.append(df.dusk.to_numpy())
            dawn.append(df.dawn.to_numpy())

        inputs = np.stack(inputs, axis=0)
        targets = np.stack(targets, axis=0)
        env = np.stack(env, axis=0)
        night = np.stack(night, axis=0)
        dusk = np.stack(dusk, axis=0)
        dawn = np.stack(dawn, axis=0)


        # find timesteps where it's night for all radars
        check_all = night.all(axis=0) # day/night mask
        # find timesteps where it's night for at least one radar
        check_any = night.any(axis=0)
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

        inputs = reshape(inputs, nights, mask, self.timesteps)
        targets = reshape(targets, nights, mask, self.timesteps)
        env = reshape(env, nights, mask, self.timesteps)
        tidx = reshape(tidx, nights, mask, self.timesteps)
        # global_dusk = reshape(global_dusk, nights, mask, self.timesteps)
        local_dusk = reshape(dusk, nights, mask, self.timesteps)
        local_dawn = reshape(dawn, nights, mask, self.timesteps)


        # create graph data objects per night
        data_list = [Data(x=torch.tensor(inputs[:, :, nidx], dtype=torch.float),
                          y=torch.tensor(targets[:, :, nidx], dtype=torch.float),
                          coords=torch.tensor(coords, dtype=torch.float),
                          areas=torch.tensor(areas, dtype=torch.float),
                          env=torch.tensor(env[..., nidx], dtype=torch.float),
                          edge_index=edge_index,
                          edge_attr=torch.stack([
                              torch.tensor(distances, dtype=torch.float),
                              torch.tensor(angles, dtype=torch.float)
                          ], dim=1),
                          tidx=torch.tensor(tidx[:, nidx], dtype=torch.long),
                          # global_dusk=torch.tensor(global_dusk[:, nidx], dtype=torch.bool),
                          local_dusk=torch.tensor(local_dusk[:, :, nidx], dtype=torch.bool),
                          local_dawn=torch.tensor(local_dawn[:, :, nidx], dtype=torch.bool))
                     for nidx in range(inputs.shape[-1])]

        # write data to disk
        os.makedirs(self.processed_dir, exist_ok=True)

        info = {'radars': voronoi.radar.values,
                 'timepoints': time,
                 'time_mask': mask,
                 'tidx': tidx,
                 'nights': nights,
                 'local_nights': night,
                 'bird_scale': self.bird_scale,
                 'boundaries': voronoi['boundary'].to_dict()}

        with open(osp.join(self.processed_dir, self.info_file_name), 'wb') as f:
            pickle.dump(info, f)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


