#!/usr/bin/env Rscript
# Title     : VPTS2VPI
# Objective : load vertical profile time series from disk, apply vertical integration, and save as netcdf
# Created by: fiona
# Created on: 18-03-22

args = commandArgs(trailingOnly=TRUE)

require(lubridate)
require(bioRad)
require(yaml)
require(stringr)
require(ncdf4)

root <- args[1]
filepath <- args[2]

config = yaml.load_file(file.path(root, "vpi_config.yml"))
sdvvp_config = yaml.load_file(file.path(root, "sdvvp_config.yml"))

begin <- as.POSIXct(config$ts, tz = "UTC")
end <- as.POSIXct(config$te, tz = "UTC")
time_range <- interval(begin, end)

# load vpts data
print(filepath)
vpts <- readRDS(filepath)
vpts <- vpts[vpts$datetime %within% time_range]

# get radar info
radar_odim_format <- tolower(str_remove(vpts$radar, "/"))
radar_alt <- vpts$attributes$where$height
lat <- vpts$attributes$where$lat
lon <- vpts$attributes$where$lon

# make a subselection for night time only
if(config$night_only){
  index_night <- check_night(vpts)
  vpts <- vpts[index_night]
}
if(config$regularize){
  vpts <- regularize_vpts(vpts)
}

# vertical integration
vpi <- integrate_profile(vpts, alt_min=(radar_alt+config$alt_min), alt_max=config$alt_max)

#define dimensions
time_dim <- ncdim_def("time", "seconds since 1970-01-01 00:00:00",
                      as.numeric(c(vpi$datetime)), unlim=FALSE)
lat_dim <- ncdim_def("lat", "degrees north", c(lat), unlim=FALSE)
lon_dim <- ncdim_def("lon", "degrees south", c(lon), unlim=FALSE)

# create netcdf file
fname <- paste0("vpi_", radar_odim_format, ".nc")
ncpath <- file.path(root, fname)

# define variables
fillvalue <- 1e32
var_def_list <- list()
for (var in attributes(vpi)$names){
  if (var != "datetime"){
    var_def_list[[var]] <- ncvar_def(var, "", list(lat_dim, lon_dim, time_dim), fillvalue, var)
  }
}
ncout <- nc_create(ncpath, var_def_list, force_v4=F)
for (var in names(var_def_list)){
    ncvar_put(ncout, var_def_list[[var]], vpi[[var]])
}

# add global attributes
ncatt_put(ncout, 0, "source", radar_odim_format)
ncatt_put(ncout, 0, "latitude", lat)
ncatt_put(ncout, 0, "longitude", lon)
ncatt_put(ncout, 0, "fillvalue", fillvalue)
ncatt_put(ncout, 0, "history", paste("F. Lippert", date(), sep=", "))

# close the file, writing data to disk
nc_close(ncout)
print('done')
