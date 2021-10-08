from birds import abm, spatial, datahandling
import os
import os.path as osp
import glob
import pickle5 as pickle
import numpy as np
import argparse
import geopandas as gpd

parser = argparse.ArgumentParser(description='process ABM simulation results')
parser.add_argument('--root', type=str, default='/home/fiona/birdMigration/data', help='entry point to required data')
parser.add_argument('--year', type=int, default=2015, help='year to be processed')
#parser.add_argument('--radar_year', type=int, default=2015, help='year to use to load radar locations and names')
parser.add_argument('--season', type=str, default='fall', help='season to be processed')
parser.add_argument('--ndummy', type=int, default=30, help='number of dummy radars')
args = parser.parse_args()

#radar_path = osp.join(args.root, 'raw', 'radar', args.season, str(args.radar_year))
abm_path = osp.join(args.root, 'raw', 'abm', args.season, str(args.year))

#radars = datahandling.load_radars(radar_path)
df = gpd.read_file(osp.join(args.root, 'raw', 'abm', 'all_radars.shp'))
radars = dict(zip(zip(df.lon, df.lat), df.radar.values))
#radar_index = {name : idx for idx, name in enumerate(radars.values())}
#radar_index['sink'] = len(radar_index)
#N = len(radar_index)

sp = spatial.Spatial(radars, n_dummy_radars=args.ndummy)
cells, G = sp.voronoi()
cells = cells.to_crs(f'epsg:{sp.epsg_lonlat}')   #to_crs(epsg='4326')

radar_index = {name : idx for idx, name in enumerate(cells.radar.values)}
N = len(radar_index)

traj = np.load(osp.join(abm_path, 'traj.npy'))
states = np.load(osp.join(abm_path, 'states.npy'))
T = states.shape[0]
with open(osp.join(abm_path, 'time.pkl'), 'rb') as f:
    time = pickle.load(f)


departing = np.zeros((T, N))
landing = np.zeros((T, N))
outfluxes = np.zeros((T, N, N))

for tidx in range(T):
    print(f'computing fluxes for time step {tidx}')

    df_flows = abm.bird_fluxes(traj, states, tidx, cells)

    if len(df_flows) > 0:
        groups = df_flows.groupby('radar')
        #grouped = groups['dst_radar'].value_counts()

        for src, df_dst in groups:
            fluxes = df_dst['dst_radar'].value_counts()
            for dst, count in fluxes.items():
                outfluxes[tidx, radar_index[src], radar_index[dst]] = count

    # count departing and landing birds
    departing_t = abm.departing_birds(traj, states, tidx, cells)
    landing_t = abm.landing_birds(traj, states, tidx, cells)

    if len(departing_t) > 0:
        groups = departing_t.groupby('radar')
        for radar, df_radar in groups:
            departing[tidx, radar_index[radar]] = len(df_radar)

    if len(landing_t) > 0:
        groups = landing_t.groupby('radar')
        for radar, df_radar in groups:
            landing[tidx, radar_index[radar]] = len(df_radar)

np.save(osp.join(abm_path, f'outfluxes.npy'), outfluxes)
np.save(osp.join(abm_path, 'departing_birds.npy'), departing)
np.save(osp.join(abm_path, 'landing_birds.npy'), landing)



