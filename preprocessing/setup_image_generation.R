require(bioRad)
require(uvaRadar)
require(stars)
require(yaml)

config = yaml.load_file("config.yml")

# set credentials for UvA Radar Data Storage
s3_set_key(username = config$login$username,
           password = config$login$password)

bbox <- st_bbox(get_radars_df(config$radars)$geometry)
config[['bbox']] <- bbox

f(!dir.exists(config$data$tiff)){
  dir.create(config$data$tiff, recursive=TRUE)
}
subdir <- file.path(config$data$tiff, paste(config$ts, "-", config$te))
if(!dir.exists(subdir)){
  dir.create(subdir)
}

config_file <- file(file.path(subdir, "config.yml"), open="w")
write_yaml(config, config_file)
