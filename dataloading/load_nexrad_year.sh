#!/bin/bash

YEAR=$1
# make new directory
rclone mkdir box:my_nexrad_vpts:$YEAR
# copy data to private folder
rclone copy --include "*$YEAR*" box:nexrad_vpts box:my_nexrad_vpts:$YEAR