from birds import abm, spatial, datahandling
import os
import os.path as osp
import glob
import pickle5 as pickle
import numpy as np

radar_path = '/home/fiona/birdMigration/data/raw/radar/fall/2015'
abm_path = '/home/fiona/birdMigration/data/raw/abm/fall/2015'
radars = datahandling.load_radars(radar_path)
radar_index = {name : idx for idx, name in enumerate(radars.values())}
radar_index['sink'] = len(radar_index)
N = len(radar_index)

out_dir = osp.join(abm_path, 'outfluxes_abs')
os.makedirs(out_dir, exist_ok=True)

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

sp = spatial.Spatial(radars)
cells = sp.voronoi_with_sink().to_crs(epsg='4326')

traj, states, time = load_sim_results(abm_path)
T = traj.shape[0]

departing = np.zeros((T, N))
landing = np.zeros((T, N))

for tidx in range(T):
    print(f'computing flows for time step {tidx}')
    # compute flows from time tidx to tidx + 1
    # flows = abm.bird_flows(traj, states, tidx, cells)
    #
    # outfluxes = np.zeros((N, N))
    #
    # if len(flows) > 0:
    #     groups = flows.groupby('radar')
    #     grouped = groups['dst_radar'].value_counts()
    #
    #     for src, df_dst in groups:
    #         fluxes = df_dst['dst_radar'].value_counts()
    #         z = fluxes.sum()
    #         for dst, count in fluxes.items():
    #             outfluxes[radar_index[src], radar_index[dst]] = count #/ z
    #
    # np.save(osp.join(out_dir, f'{tidx}.npy'), outfluxes)

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


np.save(osp.join(abm_path, 'departing_birds.npy'), departing)
np.save(osp.join(abm_path, 'landing_birds.npy'), landing)



