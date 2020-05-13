#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

require(bioRad)
require(uvaRadar)
require(stars)
require(yaml)

config_file <- file.path(args[1], "config.yml")
config = yaml.load_file(config_file) #"config.yml")

# set credentials for UvA Radar Data Storage
s3_set_key(username = config$login$username,
           password = config$login$password)

bbox <- st_bbox(get_radars_df(config$radars)$geometry)
#config[['bounds']] <- bbox
config[['lon_min']] <- bbox[[1]] - config$reach
config[['lat_min']] <- bbox[[2]] - config$reach
config[['lon_max']] <- bbox[[3]] + config$reach
config[['lat_max']] <- bbox[[4]] + config$reach

write_yaml(config, config_file)
