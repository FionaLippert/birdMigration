from pygam import PoissonGAM, te
import numpy as np
from birds import datahandling, abm, spatial, era5interface, datasets
import os.path as osp
import os
import argparse
import geopandas as gpd
import pandas as pd
import pickle5 as pickle
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='GAM baseline')
parser.add_argument('--data_root', type=str, default='/home/fiona/birdMigration/data', help='entry point to required data')
parser.add_argument('--data_source', type=str, default='abm', help='data source to be used')
parser.add_argument('--season', type=str, default='fall', help='season; either fall or spring')
parser.add_argument('--bird_scale', type=int, default=2000, help='scalar by which absolute bird counts will be normalized by')
args = parser.parse_args()

def persistence(last_ob, timesteps):
    # always return last observed value
	return [last_ob] * timesteps


def fit_baseGAM(vid, day_of_year, solarpos, solarpos_change):
    gam = PoissonGAM(te(0, 1, 2)) # poisson distribution and log link
    gam.fit(np.stack([day_of_year, solarpos, solarpos_change], axis=1), vid)
    return gam

def predict_baseGAM(gam, day_of_year, solarpos, solarpos_change):
    features = np.stack([day_of_year, solarpos, solarpos_change], axis=1)
    y = gam.predict(features)
    return y

def fit_envGAM(vid, baseGAM_pred, wind_speed, wind_dir):
    gam = PoissonGAM(te(0, 1, 2)) # poisson distribution and log link
    gam.fit(np.stack([baseGAM_pred, wind_speed, wind_dir], axis=1), vid)
    return gam

def predict_envGAM(gam, baseGAM_pred, wind_speed, wind_dir):
    features = np.stack([baseGAM_pred, wind_speed, wind_dir], axis=1)
    y = gam.predict(features)
    return y



#root = '/home/fiona/birdMigration/data'
root = args.data_root
raw_dir = osp.join(root, 'raw')
radar_dir = osp.join(raw_dir, 'radar')
abm_dir = osp.join(raw_dir, 'abm')
env_dir = osp.join(raw_dir, 'env')
season = args.season #'fall'
data_source = args.data_source #'abm'
bird_scale = args.bird_scale #2000
#load_baseGAM = True
csv_file = osp.join(root, 'seasonal_trends', f'gam_summary_{data_source}.csv')

if data_source == 'abm':
    years = ['2015', '2016', '2017', '2018', '2019']
else:
    years = ['2015', '2016', '2017']

df = []
for year in years:

    preprocessed_dir = osp.join(root, 'preprocessed', data_source, season, year)
    feature_path = osp.join(preprocessed_dir, 'dynamic_features.csv')
    if not osp.isfile(feature_path):
        os.makedirs(preprocessed_dir, exist_ok=True)
        datasets.prepare_features(preprocessed_dir, raw_dir, data_source, season, year)

    dynamic_feature_df = pd.read_csv(feature_path)
    #voronoi = gpd.read_file(osp.join(preprocessed_dir, 'voronoi.shp'))

    df.append(dynamic_feature_df)

df = pd.concat(df, ignore_index=True)
df['dayofyear'] = pd.DatetimeIndex(df.datetime).dayofyear
df.radar.replace(['nlhrw', 'nldbl'], 'nlhrw-nldbl', inplace=True)
print(df.isna().size)
df.fillna(0, inplace=True)

df_new = []
for r in df.radar.unique():
    print(r)
    ridx = df.radar == r
    dfr = df[ridx].dropna()
    gam = fit_baseGAM(dfr['birds'] / bird_scale,
                      dfr['dayofyear'],
                      dfr['solarpos'],
                      dfr['solarpos_dt'])

    with open(osp.join(root, 'seasonal_trends', f'gam_base_model_{data_source}_{r}.pkl'), 'wb') as f:
        pickle.dump(gam, f)

    y_gam = predict_baseGAM(gam,
                            df.loc[ridx, 'dayofyear'],
                            df.loc[ridx, 'solarpos'],
                            df.loc[ridx, 'solarpos_dt'])

    df.loc[ridx, 'gam_pred'] = y_gam * bird_scale
    #df_new.append(df_radar)

#df_new = pd.concat(df_new, ignore_index=True)
df.to_csv(osp.join(root, 'seasonal_trends', f'gam_summary_{data_source}.csv'))

assert 0




all_data = []
all_days = []
all_solarpos = []
all_solarpos_change = []

radars = datahandling.load_radars(osp.join(radar_dir, season, '2015'))
cells = spatial.Spatial(radars).cells

if load_baseGAM:
    df = pd.read_csv(csv_file)
dfs_env = []
dfs = []
for coords, name in radars.items():
    vid = []
    days = []
    sun = []
    sun_change = []
    years = []
    names = []
    time = []
    wind_speed = []
    wind_dir = []
    print(name)
    for y in range(2015, 2020):
        year = str(y)
        if data_source == 'radar':
            if y > 2016 and name == 'nldbl':
                name = 'nlhrw'
            data, _, t_range = datahandling.load_season(radar_dir, season, year, 'vid',
                                                         mask_days=True, radar_names=[name])
            t_range = t_range.tz_localize('UTC')
        else:
            radar_cell = cells[cells.radar == name]
            data, t_range = abm.load_season(abm_dir, season, year, radar_cell)

        # load solar position and compute relative changes
        t_range_sun = t_range.insert(-1, t_range[-1] + pd.Timedelta(t_range.freq))
        solarpos = datahandling.get_solarpos(t_range_sun, [coords]).flatten()
        solarpos_change = solarpos[:-1] - solarpos[1:]

        wind = era5interface.extract_points(
            os.path.join(env_dir, season, year, 'pressure_level_850.nc'),
            [coords], t_range.tz_localize(None), vars=['u', 'v'])

        speed = np.sqrt(np.square(wind['u']) + np.square(wind['v']))
        dir = (abm.uv2deg(wind['u'], wind['v']) + 360) % 360

        # set bird counts to zero during the day
        data[~np.isfinite(data)] = 0

        vid.append(data.flatten())
        days.append(t_range.dayofyear)
        sun.append(solarpos[:-1])
        sun_change.append(solarpos_change)
        years.append(np.ones(solarpos_change.size)*y)
        names.append(np.array([name]*solarpos_change.size))
        time.append(t_range)
        wind_speed.append(speed)
        wind_dir.append(dir)

    vid_test = vid[0] / bird_scale
    wind_speed_test = wind_speed[0]
    wind_dir_test = wind_dir[0]
    names_test = names[0]
    time_test = time[0]
    vid_train = np.concatenate(vid[1:]) / bird_scale
    wind_speed_train = np.concatenate(wind_speed[1:])
    wind_dir_train = np.concatenate(wind_dir[1:])


    vid = np.concatenate(vid) / bird_scale
    days = np.concatenate(days)
    sun = np.concatenate(sun)
    sun_change = np.concatenate(sun_change)
    years = np.concatenate(years)
    names = np.concatenate(names)
    time = np.concatenate(time)


    if load_baseGAM:
        y_gam = df[df.radar == name].gam_prediction.to_numpy()
        print('base GAM loaded')
    else:
        gam = fit_baseGAM(vid, days, sun, sun_change)
        with open(osp.join(root, 'seasonal_trends', f'gam_base_model_{data_source}_{name}.pkl'), 'wb') as f:
            pickle.dump(gam, f)

        y_gam = predict_baseGAM(gam, days, sun, sun_change)

        dfs.append(pd.DataFrame({'radar': names,
                                 'year': years,
                                 'dayofyear': days,
                                 'datetime': time,
                                 'solarposition': sun,
                                 'solarchange': sun_change,
                                 'vid': vid * bird_scale,
                                 'gam_prediction': y_gam * bird_scale}))

    y_gam_train = dfs[-1][dfs[-1].year > 2015].gam_prediction
    y_gam_test = dfs[-1][dfs[-1].year == 2015].gam_prediction
    gam_env = fit_envGAM(vid_train, y_gam_train, wind_speed_train, wind_dir_train)
    with open(osp.join(root, 'seasonal_trends', f'gam_env_model_{data_source}.pkl'), 'wb') as f:
        pickle.dump(gam_env, f)
    y_gam_env = predict_envGAM(y_gam_test, wind_speed_test, wind_dir_test)

    dfs_env.append(pd.DataFrame({'radar': names_test,
                             'datetime': time_test,
                             'vid': vid_test * bird_scale,
                             'gam_prediction': y_gam_env * bird_scale}))


if not load_baseGAM:
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(osp.join(root, 'seasonal_trends', f'gam_summary_{data_source}.csv'))

df_env = pd.concat(dfs_env, ignore_index=True)
df_env.to_csv(osp.join(root, 'seasonal_trends', f'gam_env_model_summary_{data_source}.csv'))




