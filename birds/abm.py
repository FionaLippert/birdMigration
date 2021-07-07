import numpy as np
from tqdm import tqdm
import geopy
from geopy.distance import distance
from shapely import geometry
from pvlib import solarposition
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt
import os
import os.path as osp
import pickle5 as pickle
from datetime import datetime
import xarray as xr
import glob

from birds import datahandling
from birds import spatial
from birds.era5interface import ERA5Loader

class Environment:
    def __init__(self, wind, freq='1H'):
        self.bounds = geometry.Polygon([(wind.longitude.max(), wind.latitude.max()),
                                        (wind.longitude.max(), wind.latitude.min()),
                                        (wind.longitude.min(), wind.latitude.min()),
                                        (wind.longitude.min(), wind.latitude.max())])
        self.wind = wind
        self.time = extract_time(wind, freq, 'UTC')  # pandas datetimeindex
        #self.dt = pd.to_timedelta(self.time.freq).total_seconds()  # time step for simulation
        self.dt = pd.Timedelta(freq).total_seconds()  # time step for simulation

    def get_wind(self, tidx, lon, lat, pref_dir):
        # load wind data at (lon, lat) using linear interpolation
        wind_t = self.wind.isel(time=tidx).interp(longitude=lon, latitude=lat, method='linear')
        wind_speed = float(np.sqrt(wind_t.u**2 + wind_t.v**2))
        wind_dir = np.deg2rad((float(uv2deg(wind_t.u, wind_t.v)) - pref_dir + 360) % 360)
        return wind_speed, wind_dir

    def get_sun(self, tidx, lon, lat):
        # compute solar position
        tidx = min(tidx, len(self.time)-1)
        time = self.time[tidx]
        sun = float(solarposition.get_solarposition(time, lat, lon).elevation)
        return sun

class Bird:
    def __init__(self, id, lat, lon, env, start_day,
                 endogenous_heading=215, pref_dir_std=0,
                 air_speed=10, compensation=0.5, energy_tol=0,
                 departure_window=1, target_lon=None, target_lat=None):

        # bird and system properties
        self.id = id
        self.env = env
        self.start_day = start_day
        self.endogenous_heading = endogenous_heading # clockwise from north
        self.pref_dir_std = pref_dir_std
        self.air_speed = air_speed # in m/s
        self.compensation = compensation
        self.energy_tol = energy_tol # if <= 0 no headwinds are tolerated
        self.departure_window = departure_window # number of timesteps after dusk within which birds can depart
        self.target_pos = (target_lon, target_lat)

        # initialize simulation
        self.reset(lat, lon)

    def reset(self, lat, lon):
        self.pos = geopy.Point(latitude=lat, longitude=lon)
        self.state = 0  # landed (one of [1: 'flying', 0: 'landed', -1: 'exited']
        self.tidx = 0
        self.migrating = False
        self.ground_speed = 0
        self.night_count = 0
        self.dir_north = 0
        self.sample_pref_dir()


    def step(self):

        if self.state == 1:
            # previously flying, so compute new position
            dist = distance(meters=self.ground_speed * self.env.dt)
            self.pos = dist.destination(point=self.pos, bearing=self.dir_north)

        if self.check_bounds():
            if self.check_night():
                # current conditions
                wind_speed, wind_dir = self.env.get_wind(self.tidx, self.pos.longitude,
                                                         self.pos.latitude, self.pref_dir)
                self.adjust_heading(wind_speed, wind_dir)
                self.compute_drift(wind_speed, wind_dir)
                self.compute_ground_speed(wind_speed, wind_dir)

                # check if weather conditions are good enough to start/continue migrating
                self.compute_energy()
                fly = self.check_departure(wind_speed, wind_dir)
                depart = self.night_count < self.departure_window and self.state == 0 and fly
                keep_flying = self.state == 1 and fly
                if depart or keep_flying:
                    self.state = 1
                    # compute flight direction
                    self.dir_north = self.pref_dir + np.rad2deg(self.drift)
                elif self.state == 1 and not fly:
                    # land because wind conditions are not suitable anymore
                    self.state = 0
                    # determine new preferred migration direction for next departure
                    self.sample_pref_dir()

                self.night_count += 1

            else:
                if self.state == 1:
                    # land because end of the night has been reached
                    self.state = 0
                    # determine new preferred migration direction for next departure
                    self.sample_pref_dir()
                self.night_count = 0

        else:
            # left simulated region
            self.state = -1

        self.tidx += 1

    def check_bounds(self):
        return self.env.bounds.contains(geometry.Point(self.pos.longitude, self.pos.latitude))

    def compute_pref_dir(self):

        x = self.target_pos[0] - self.pos.longitude
        y = self.target_pos[1] - self.pos.latitude

        rad = np.arctan2(x, y)
        deg = np.rad2deg(rad)

        # make sure angle is between 0 and 360 degree
        deg = (deg + 180) % 360 + 180
        return deg

    def sample_pref_dir(self):
        #self.pref_dir = np.random.normal(self.endogenous_heading, self.pref_dir_std)
        self.pref_dir = np.random.normal(self.compute_pref_dir(), self.pref_dir_std)

    def adjust_heading(self, wind_speed, wind_dir):
        # compute heading based on given relative wind compensation
        # if desired compensation is not possible, choose heading perpendicular to pref_dir to compensate
        # as much as possible
        self.heading = - np.arcsin(np.clip(self.compensation * wind_speed * np.sin(wind_dir) / self.air_speed, -1, 1))

    def compute_drift(self, wind_speed, wind_dir):
        # drift relative to pref_dir
        self.drift = np.arctan((self.air_speed * np.sin(self.heading) + wind_speed * np.sin(wind_dir)) /
                               (self.air_speed * np.cos(self.heading) + wind_speed * np.cos(wind_dir)))

    def compute_ground_speed(self, wind_speed, wind_dir):
        self.ground_speed = self.air_speed * np.cos(self.heading - self.drift) + \
                            wind_speed * np.cos(wind_dir - self.drift)

    def compute_energy(self):
        # energy expenditure per unit distance travelled along the preferred direction
        # relative to optimal energy expenditure (= air_speed)
        # "transport cost" see McLaren 2012 and Pennycuick 2008
        if self.ground_speed <= 0:
            # bird is blown in opposite direction by wind
            self.energy = np.inf
        else:
            self.energy = self.air_speed / (self.ground_speed * np.cos(self.drift)) - 1

    def check_night(self):
        # check if hour falls into the night, or includes dusk or dawn
        sun_start = self.env.get_sun(self.tidx, self.pos.longitude, self.pos.latitude)
        sun_end = self.env.get_sun(self.tidx+1, self.pos.longitude, self.pos.latitude)
        return sun_start < -6 or sun_end < -6

    def check_departure(self, wind_speed, wind_dir):
        # decision for departure/landing
        # course can only be maintained if airspeed is faster than the wind component
        # perpendicular to the direction of travel

        if not self.migrating:
            #print(self.start_day, self.time[self.tidx].day)
            dt = (self.env.time[self.tidx] - self.env.time[0]).days
            self.migrating = (self.start_day <= dt)

        check_speed = self.air_speed >= self.compensation * wind_speed * np.sin(wind_dir) # - self.drift)
        #check_drift = np.abs(self.drift) <= self.drift_tol
        check_energy = self.energy <= self.energy_tol

        if check_speed and check_energy and self.migrating:
            return True
        else:
            return False

class DataCollection:
    def __init__(self, time, num_birds, settings):
        self.num_birds = num_birds
        self.time = time
        self.T = len(time)
        #self.buffers = buffers # shapely polygons with (lon, lat) coords
        self.settings = settings

        self.clear_data()

    def clear_data(self):
        self.data = {
                     'trajectories': np.zeros((self.T, self.num_birds, 2), dtype=np.float),
                     'states': np.zeros((self.T, self.num_birds), dtype=np.int),
                     #'counts': np.zeros((self.T, len(self.buffers)), dtype=np.long),
                     #'directions': np.zeros((self.T, len(self.buffers)), dtype=np.long),
                     'ground_speeds': np.zeros((self.T, self.num_birds), dtype=np.long),
                     'directions': np.zeros((self.T, self.num_birds), dtype=np.long)
                     }

    def collect(self, tidx, birds):
        assert(len(birds) == self.num_birds)
        for bird in birds:
            self.data['trajectories'][tidx, bird.id] = [bird.pos.longitude, bird.pos.latitude]
            self.data['states'][tidx, bird.id] = bird.state
            self.data['ground_speeds'][tidx, bird.id] = bird.ground_speed
            self.data['directions'][tidx, bird.id] = bird.dir_north

            # if bird.state == 1:
            #     # flying
            #     pt = geometry.Point([bird.pos.longitude, bird.pos.latitude])
            #     for bidx, b in self.buffers.items():
            #         if b.contains(pt):
            #             self.data['counts'][bird.tidx, bidx] += 1
            #             self.data['directions'][bird.tidx, bidx] += rad2deg(bird.dir)
            #             break

    def save(self, file_path):
        self.data['time'] = self.time
        self.data['settings'] = self.settings
        self.data['last_modified'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        # write to disk
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(self.data, f, pickle.HIGHEST_PROTOCOL)


class Simulation:
    def __init__(self, env, settings, **kwargs):
        self.settings = settings
        self.env = env
        self.rng = np.random.default_rng(settings['random_seed'])

        for k in kwargs.keys():
            if k in ['departure_area', 'target_area']:
                self.__setattr__(k, kwargs[k])

        self.spawn_birds()
        self.data = DataCollection(env.time, len(self.birds), settings)


    def spawn_birds(self):
        # initialize birds
        self.birds = []

        for id in range(self.settings['num_birds']):
            # sample initial position of bird
            print(self.settings)
            if 'start_line' in self.settings:
                lon, lat = self.sample_pos_from_line()
                print(lon, lat)
            elif hasattr(self, 'departure_area'):
                lon, lat = self.sample_pos(self.departure_area)
                #print('sampling from departure_area was successful')
            elif 'sources' in self.settings:
                #print('sampling from sources')
                source = self.rng.choice(self.settings['sources'])
                lon = self.rng.normal(source[0], self.settings['source_std'])
                lat = self.rng.normal(source[1], self.settings['source_std'])
            else:
                #print('sampling uniformly')
                minx, miny, maxx, maxy = self.env.bounds.buffer(-1e-10).bounds
                if self.rng.random() > 0.5:
                    lat = maxy
                    lon = self.rng.uniform(minx, maxx)
                else:
                    lat = self.rng.uniform(miny, maxy)
                    lon = maxx

            if hasattr(self, 'target_area'):
                target_lon, target_lat = self.sample_target_pos()
                print('target_pos', target_lon, target_lat)

            # start_day = self.rng.normal(self.settings['start_day_mean'], self.settings['start_day_std'])
            start_day = self.rng.choice(range(self.settings['start_day_range']))
            energy_tol = self.rng.normal(self.settings['energy_tol_mean'], self.settings['energy_tol_std'])
            self.birds.append(Bird(id, lat, lon, self.env, start_day,
                                   compensation=self.settings['compensation'],
                                   departure_window=self.settings['departure_window'],
                                   energy_tol=energy_tol, pref_dir_std=self.settings['pref_dir_std'],
                                   target_lon=target_lon, target_lat=target_lat))

    def sample_pos(self, area):
        minx, miny, maxx, maxy = area.total_bounds
        lon = self.rng.uniform(minx, maxx)
        lat = self.rng.uniform(miny, maxy)
        pos = geometry.Point(lon, lat)
        while not area.contains(pos).any():
            lon = np.random.uniform(minx, maxx)
            lat = np.random.uniform(miny, maxy)
            pos = geometry.Point(lon, lat)
        return lon, lat

    def sample_target_pos(self):
        minx, miny, maxx, maxy = self.target_area.total_bounds
        print(minx, miny, maxx, maxy)
        lon = self.rng.uniform(minx, maxx)
        lat = self.rng.uniform(miny, maxy)
        pos = geometry.Point(lon, lat)
        while not self.target_area.contains(pos).any():
            lon = np.random.uniform(minx, maxx)
            lat = np.random.uniform(miny, maxy)
            pos = geometry.Point(lon, lat)
        return lon, lat

    def sample_pos_from_line(self, n_options=1000):
        bounds = gpd.GeoSeries([self.env.bounds.buffer(-1e-10)], crs='EPSG:4326')
        p1, p2 = self.settings['start_line']
        line = gpd.GeoSeries([geometry.LineString([geometry.Point(*p1),
                                                           geometry.Point(*p2)])], crs='EPSG:4326')
        line = line.intersection(bounds)
        n = self.rng.uniform(0, n_options + 1)
        pos = line.interpolate(n / n_options, normalized=True)
        lon = float(pos.x)
        lat = float(pos.y)

        return lon, lat

    def run(self, steps):
        for tidx in tqdm(range(steps)):
            for bird in self.birds:
                bird.step()
            self.data.collect(tidx, self.birds)

    def reset(self):
        for bird in self.birds:
            lon0, lat0 = self.data.data['trajectories'][0, bird.id]
            bird.reset(lon0, lat0)
        self.data.clear_data()

    def save_data(self, file_path):
        self.data.save(file_path)


def uv2deg(u, v):
    # v and u wind components to direction into which wind is blowing (opposite of meteorological direction!)
    deg = ((180 * np.arctan2(u, v) / np.pi) + 360) % 360
    return deg

def deg2uv(deg, speed):
    # v and u wind components to direction into which wind is blowing (opposite of meteorological direction!)
    # degree is relative to 0 deg north
    u = speed * np.sin(np.deg2rad(deg))
    v = speed * np.cos(np.deg2rad(deg))
    return u, v


def rad2deg(rad):
    rad = (rad + np.pi) % np.pi
    deg = np.rad2deg(rad)
    return deg

def extract_time(xr_dataset, freq, tz):
    time = pd.to_datetime(xr_dataset.time.values)
    time = pd.date_range(time[0], time[-1], freq=freq)
    time = time.tz_localize(tz=tz, ambiguous=False)
    return time

def plot_trajectories(birds, filename):
    fig, ax = plt.subplots()
    for bird in birds:
        xx, xy = zip(*bird.trajectory)
        #print(bird.states)
        lidx = np.where(np.array(bird.states) == 0)
        traj = ax.plot(xx, xy)
        color = traj[0].get_color()
        ax.plot(xx[0], xy[0], 'o', c='red')
        ax.scatter(np.array(xx)[lidx], np.array(xy)[lidx], facecolors='none', edgecolors=color, alpha=0.1)
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def bird_counts(birds, timesteps, minx, miny, maxx, maxy):
    gridx = np.arange(np.ceil(minx), np.ceil(maxx)+1, 1)
    gridy = np.arange(np.ceil(miny), np.ceil(maxy)+1, 1)
    counts = np.zeros((timesteps, gridx.size, gridy.size))

    for bird in birds:
        xx = np.digitize(np.array(bird.trajectory)[:, 0], gridx)
        yy = np.digitize(np.array(bird.trajectory)[:, 1], gridy)
        fidx = np.where(np.array(bird.states) == 1)
        for t in fidx[0]:
            counts[t, xx[t], yy[t]] += 1

    return counts


def make_grid(extent=[0.36, 46.36, 16.07, 55.40], res=0.5, crs='4326'):
    xmin, ymin, xmax, ymax = extent
    cols = np.arange(int(np.floor(xmin))-1, int(np.ceil(xmax))+1, res)
    rows = np.arange(int(np.floor(ymin))-1, int(np.ceil(ymax))+1, res)
    rows = rows[::-1]
    polygons = []
    for x in cols:
        for y in rows:
            polygons.append(geometry.Polygon([(x,y), (x+res, y), (x+res, y-res), (x, y-res)]))

    grid = gpd.GeoDataFrame({'geometry': polygons}, crs=f'epsg:{crs}')
    return grid

def get_points(trajectories, states, state=1, vars={}):
    df = gpd.GeoDataFrame({'geometry': []}, crs='epsg:4326')
    mask = np.where(states == state)
    if len(mask[0]) > 0:
        xx = trajectories[mask, 0].flatten()
        yy = trajectories[mask, 1].flatten()
        df['geometry'] = gpd.points_from_xy(xx, yy)
        for k, v in vars.items():
            # add additional variables to dataframe
            df[k] = v[mask]
    return df


def aggregate(trajectories, states, grid, t_range, state):
    names = []
    grid_counts = grid.to_crs('epsg:4326')    # to lonlat crs
    for t in t_range:
        name_t = f'n_birds_{t}'
        df_t = get_points(trajectories[t], states[t], state)
        if len(df_t) > 0:
            merged = gpd.sjoin(df_t, grid_counts, how='left', op='within')
            merged[f'n_birds_{t}'] = 1
            dissolve = merged.dissolve(by="index_right", aggfunc="count")
            grid_counts.loc[dissolve.index, name_t] = dissolve[name_t].values
            grid_counts[name_t].fillna(0, inplace=True)
        else:
            # no birds found
            grid_counts[name_t] = 0
        names.append(name_t)
    return grid_counts, names

def aggregate_uv(trajectories, states, grid, t_range, state, u, v):
    cols_u = []
    cols_v = []
    grid_df = grid.to_crs('epsg:4326')    # to lonlat crs
    for t in t_range:
        df_t = get_points(trajectories[t], states[t], state, {'u': u, 'v': v})
        cols_u.append(f'u_{t}')
        cols_v.append(f'v_{t}')
        if len(df_t) > 0:
            merged = gpd.sjoin(df_t, grid_df, how='left', op='within')
            dissolve = merged.dissolve(by="index_right", aggfunc="mean")
            grid_df.loc[dissolve.index, cols_u[-1]] = dissolve[cols_u[-1]].values
            grid_df.loc[dissolve.index, cols_v[-1]] = dissolve[cols_v[-1]].values
            grid_df[[cols_u[-1], cols_v[-1]]].fillna(0, inplace=True)
        else:
            # no birds found
            grid_df[[cols_u[-1], cols_v[-1]]] = 0
    return grid_df, cols_u, cols_v


def bird_flows(trajectories, states, tidx, grid):
    mask = np.where(states[tidx] == 1)
    df_t0 = gpd.GeoDataFrame({'geometry': []}, crs='epsg:4326')
    df_t1 = gpd.GeoDataFrame({'geometry': []}, crs='epsg:4326')
    if len(mask[0]) > 0:
        # get grid cell of all flying birds at timestep tidx
        xx_t0 = trajectories[tidx, mask, 0].flatten()
        yy_t0 = trajectories[tidx, mask, 1].flatten()
        df_t0['geometry'] = gpd.points_from_xy(xx_t0, yy_t0)
        #print('number of birds flying = ', len(df_t0))

        # get grid cell of all previously flying birds at next timestep tidx+1
        xx_t1 = trajectories[tidx+1, mask, 0].flatten()
        yy_t1 = trajectories[tidx+1, mask, 1].flatten()
        df_t1['geometry'] = gpd.points_from_xy(xx_t1, yy_t1)


    # determine flows
    merged_t0 = gpd.sjoin(df_t0, grid, how='inner', op='within')
    merged_t1 = gpd.sjoin(df_t1, grid, how='inner', op='within')
    merged_t0['dst_radar'] = merged_t1['radar']
    merged_t0['dst_index'] = merged_t1['index_right']
    return merged_t0

def count_birds_of_interest(positions, grid):
    df = gpd.GeoDataFrame({'geometry': []}, crs='epsg:4326')

    if positions.shape[1] > 0:
        # get positions of all birds of interest
        xx_t0 = positions[..., 0].flatten()
        yy_t0 = positions[..., 1].flatten()
        df['geometry'] = gpd.points_from_xy(xx_t0, yy_t0)

    # count birds of interest per grid cell
    merged = gpd.sjoin(df, grid, how='inner', op='within')
    return merged

def departing_birds(trajectories, states, tidx, grid):
    if tidx == 0:
        mask = np.where(states[tidx] == 1)
    else:
        mask = np.where(np.logical_and(states[tidx-1] == 0, states[tidx] == 1))

    departing = count_birds_of_interest(trajectories[tidx, mask], grid)
    return departing

def landing_birds(trajectories, states, tidx, grid):
    if tidx == 0:
        mask = []
    else:
        mask = np.where(np.logical_and(states[tidx-1] == 1, states[tidx] == 0))

    landing = count_birds_of_interest(trajectories[tidx, mask], grid)
    return landing




def load_season(root, season, year, cells, uv=True):

    abm_dir = osp.join(root, season, year)

    traj = np.load(osp.join(abm_dir, 'traj.npy'))
    states = np.load(osp.join(abm_dir, 'states.npy'))
    directions = np.load(osp.join(abm_dir, 'directions.npy'))
    speeds = np.load(osp.join(abm_dir, 'ground_speeds.npy'))
    T = states.shape[0]

    with open(osp.join(abm_dir, 'time.pkl'), 'rb') as f:
        time = pickle.load(f)

    counts, cols = aggregate(traj, states, cells, range(T), state=1)
    # counts = counts.fillna(0)
    data = counts[cols].to_numpy()

    if uv:
        u, v = deg2uv(directions, speeds)  # in meters
        grid_df, cols_u, cols_v = aggregate_uv(traj, states, cells, range(T), 1, u, v)
        grid_df = grid_df.fillna(0)
        u = grid_df[cols_u].to_numpy()
        v = grid_df[cols_v].to_numpy()
        return data, time, u, v
    else:
        return data, time





if __name__ == '__main__':

    year = '2015'
    season = 'fall'
    start_date = f'{year}-08-01 12:00+00:00'
    #end_date = f'{year}-10-15 12:00+00:00'
    end_date = f'{year}-09-15 12:00+00:00'
    root = '/home/fiona/birdMigration/data/raw'
    wind_path = osp.join(root, 'wind', season, year, 'wind_850.nc')
    radar_path = osp.join(root, 'radar', season, year)

    radars = datahandling.load_radars(radar_path)
    # get 25km buffers around radar stations to simulate VP measurements
    sp = spatial.Spatial(radars)
    buffers = sp.pts_local.buffer(25_000).to_crs(epsg=sp.epsg).to_dict()

    if not osp.exists(wind_path):
        radars = datahandling.load_radars(radar_path)
        spatial = spatial.Spatial(radars)
        minx, miny, maxx, maxy = spatial.cells.to_crs(epsg=spatial.epsg).total_bounds
        bounds = [maxy, minx, miny, maxx] # North, West, South, East
        ERA5Loader().download_season(year, season, wind_path, bounds)

    wind = xr.open_dataset(wind_path).sel(time=slice(start_date, end_date))
    env = Environment(wind)

    settings = {'num_birds': 2,
                'random_seed': 3,
                'start_day_mean': 2,
                'start_day_std': 5,
                'energy_tol_mean': 1.0,
                'energy_tol_std': 0.1,
                'compensation': 0.75}

    # initialize birds
    # birds = []
    # np.random.seed(settings['random_seed'])
    # for id in range(settings['num_birds']):
    #     #lat = np.random.uniform(wind.latitude.mean(), wind.latitude.max())
    #     r = np.random.rand()
    #     if r > 0.5:
    #         lat = wind.latitude.max()
    #         lon = np.random.uniform(wind.longitude.min(), wind.longitude.max())
    #     else:
    #         lat = np.random.uniform(wind.latitude.min(), wind.latitude.max())
    #         lon = wind.longitude.max()
    #     start_day = np.random.normal(settings['start_day_mean'], settings['start_day_std'])
    #     energy_tol = np.random.normal(settings['energy_tol_mean'], settings['energy_tol_std'])
    #     birds.append(Bird(id, lat, lon, env, start_day, compensation=settings['compensation'], energy_tol=energy_tol))


    # simulate bird trajectories
    # data = DataCollection(env.time, len(birds), buffers, settings)
    # for t in tqdm(env.time):
    #     data.collect(birds)
    #     for bird in birds:
    #         bird.step()
    #
    # data.save(os.path.join(root, 'abm', season, year, f'simulation_{settings["num_birds"]}.pkl'))

    sim = Simulation(env, buffers, settings)
    sim.run(len(env.time))

    # with open(os.path.join(root, 'abm', season, year, 'test_simulation.pkl'), 'rb') as f:
    #     data = pickle.load(f)
    #
    # print(data)

    sim.data.plot_trajectories(f'trajectories_{settings["num_birds"]}.png')
    #
    # counts = bird_counts(birds, T, bounds)
    # np.save('test_counts.npy', counts)
    # #print(counts)
    # fig, ax = plt.subplots()
    # ax.imshow(counts[int(T/2)])
    # fig.savefig('test_densities.png')