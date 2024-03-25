#!/bin/bash

# run as `./load_nexrad_years.sh year_min year_max ../data/raw/nexrad_rds`

YEAR_MIN=$1
YEAR_MAX=$2
DIR=$3

# loop over years
for YEAR in $(seq $YEAR_MIN $YEAR_MAX); do
  # make new directory
  mkdir -p $DIR/$YEAR
  # copy data to local folder
  echo "Loading year" $YEAR
  rclone copy --include "*$YEAR*" box.com:nexrad_vpts $DIR/$YEAR;
done
