from pygam import PoissonGAM, te
import numpy as np
from birds import datahandling, abm, spatial
import os.path as osp
import pandas as pd
import pandas as pd
import pickle5 as pickle
from matplotlib import pyplot as plt

def persistence(last_ob, timesteps):
    # always return last observed value
	return [last_ob] * timesteps


def fit_GAM(vid, day_of_year, solarpos, solarpos_change):
    gam = PoissonGAM(te(0, 1, 2)) # poisson distribution and log link
    gam.fit(np.stack([day_of_year, solarpos, solarpos_change], axis=1), vid)
    return gam

def predict_GAM(gam, day_of_year, solarpos, solarpos_change):
    features = np.stack([day_of_year, solarpos, solarpos_change], axis=1)
    y = gam.predict(features)
    return y



root = '/home/fiona/birdMigration/data'
radar_dir = osp.join(root, 'raw', 'radar')
abm_dir = osp.join(root, 'raw', 'abm')
season = 'fall'
data_source = 'abm'
bird_scale = 2000

all_data = []
all_days = []
all_solarpos = []
all_solarpos_change = []

radars = datahandling.load_radars(osp.join(radar_dir, season, '2015'))
cells = spatial.Spatial(radars).cells

dfs = []
for coords, name in radars.items():
    vid = []
    days = []
    sun = []
    sun_change = []
    years = []
    names = []

    for y in range(2017, 2019):
        year = str(y)
        if data_source == 'radar':
            if y > 2016 and name == 'nldbl':
                name = 'nlhrw'
            data, _, t_range = datahandling.load_season(radar_dir, season, year, 'vid',
                                                         mask_days=True, radar_names=[name])
            t_range = t_range.tz_localize('UTC')
        else:
            radar_cell = cells[cells.radar == name]
            print(radar_cell)
            data, t_range = abm.load_season(abm_dir, season, year, radar_cell)
            print(data)

        # load solar position and compute relative changes
        t_range_sun = t_range.insert(-1, t_range[-1] + pd.Timedelta(t_range.freq))
        solarpos = datahandling.get_solarpos(t_range_sun, [coords]).flatten()
        solarpos_change = solarpos[:-1] - solarpos[1:]

        # set bird counts to zero during the day
        data[~np.isfinite(data)] = 0

        vid.append(data.flatten())
        days.append(t_range.dayofyear)
        sun.append(solarpos[:-1])
        sun_change.append(solarpos_change)
        years.append(np.ones(solarpos_change.size)*y)
        names.append(np.array([name]*solarpos_change.size))

    vid = np.concatenate(vid) / bird_scale
    days = np.concatenate(days)
    sun = np.concatenate(sun)
    sun_change = np.concatenate(sun_change)
    years = np.concatenate(years)
    names = np.concatenate(names)

    gam = fit_GAM(vid, days, sun, sun_change)
    with open(osp.join(root, 'seasonal_trends', f'gam_model_{data_source}.pkl'), 'wb') as f:
        pickle.dump(gam, f)

    y_gam = predict_GAM(gam, days, sun, sun_change)

    dfs.append(pd.DataFrame({'radar': names,
                       'year': years,
                       'dayofyear': days,
                       'solarposition': sun,
                       'solarchange': sun_change,
                       'vid': vid * bird_scale,
                       'gam_prediction': y_gam * bird_scale}))

df = pd.concat(dfs, ignore_index=True)
df.to_csv(osp.join(root, 'seasonal_trends', f'gam_summary_{data_source}.csv'))




