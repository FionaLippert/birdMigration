import pandas as pd
import numpy as np
import os.path as osp

data_dir = '/media/fiona/Seagate Basic/PhD/paper_1/data/preprocessed/1H_voronoi_ndummy=30/abm/fall'
years = [2015, 2016, 2017]
cols = ['birds', 'birds_km2', 'birds_from_buffer', 'birds_km2_from_buffer', 'bird_u', 'bird_v']


for y in years:
    radar_df = pd.read_csv(osp.join(data_dir, str(y), 'dynamic_features_old.csv'))
    for col in cols:
        # remember missing radar observations
        radar_df[col] = radar_df.apply(lambda row: np.nan if (row.night and np.isnan(row[col]))
                                    else (0 if not row.night else row[col]), axis=1)
        radar_df['missing'] = radar_df['missing'] | radar_df[col].isna()

        # fill missing bird measurements by interpolation
        radar_df[col].fillna(0, inplace=True)

    radar_df.to_csv(osp.join(data_dir, str(y), 'dynamic_features.csv'))