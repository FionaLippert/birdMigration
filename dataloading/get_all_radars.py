import os
import argparse
import pandas as pd

from birds import datahandling


parser = argparse.ArgumentParser(description='script to extract all radar names and locations for a given directory')
parser.add_argument('data_dir', help='path to directory containing .nc files ')
parser.add_argument('output_dir', help='path to directory where data will be written to')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok = True)

radar_names = datahandling.load_radar_attributes(args.data_dir, 'source')
radar_longitudes = datahandling.load_radar_attributes(args.data_dir, 'longitude')
radar_latitudes = datahandling.load_radar_attributes(args.data_dir, 'latitude')
radar_altitudes = datahandling.load_radar_attributes(args.data_dir, 'altitude')



df_radars = pd.DataFrame(dict(radar=radar_names,
                              lon=radar_longitudes,
                              lat=radar_latitudes,
                              altitude=radar_altitudes))
print(df_radars)

df_radars.to_csv(os.path.join(args.output_dir, 'all_radars.csv'), index=False)