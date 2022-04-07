#!/bin/bash

YEAR=$1
DIR=$2
# make new directory
#rclone mkdir box.com:my_nexrad_vpts/$YEAR
mkdir -p $DIR/$YEAR
# copy data to private folder
rclone copy --include "*$YEAR*" box.com:nexrad_vpts $DIR/$YEAR
