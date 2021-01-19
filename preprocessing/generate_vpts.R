#!/usr/bin/env Rscript
# Title     : Generate VPTS
# Objective : load vertical profiles from database, apply vertical integration
# Created by: fiona
# Created on: 24-06-20

args = commandArgs(trailingOnly=TRUE)

require(bioRad)
require(uvaRadar)
require(stars)
require(yaml)
require(rhdf5)
require(stringr)
require(ncdf4)
require(sp)
require(maptools)


root <- args[1]
radar <- args[2]

config = yaml.load_file(file.path(root, "config.yml"))
sdvvp_config = yaml.load_file(file.path(root, "sdvvp_config.yml"))

# set credentials for UvA Radar Data Storage
s3_set_key(username = config$login$username,
           password = config$login$password)

print(radar)

vpts <- tryCatch(
              { get_vpts(radars = radar,
                    time = seq(from = as.POSIXct(config$ts, tz = "UTC"),
                               to = as.POSIXct(config$te, tz = "UTC"),
                               by = paste(config$tr, "mins")),
                    #with_db=T (not working for some reasaon. something wrong with keyring?)
                    )
              },
              error = function(cond){
                message(paste("error occured with ", radar))
                return(NA)
              })

# adjust sdvvp threshold
#print(paste('apply sdvvp threshold applied to ', radar, sdvvp_config$radar))
#sd_vvp_threshold(vpts) <- sdvvp_config$radar


# make a subselection for night time only
if(config$night_only){
  index_night <- check_night(vpts)
  vpts <- vpts[index_night]
}

vpts <- regularize_vpts(vpts)
vpi <- integrate_profile(vpts, alt_min=config$alt_min, alt_max=config$alt_max)

# extract radar location from vpts object
lat <- vpts$attributes$where$lat
lon <- vpts$attributes$where$lon

#plot(my_vpi, night_shade = FALSE, quantity="vid")

#define dimensions
time_dim <- ncdim_def("time", "seconds since 1970-01-01 00:00:00", as.numeric(c(vpi$datetime)), unlim=FALSE)
lat_dim <- ncdim_def("lat", "degrees north", c(lat), unlim=FALSE)
lon_dim <- ncdim_def("lon", "degrees south", c(lon), unlim=FALSE)


# create netcdf file
fname <- paste0("vpi_", str_remove(radar, "/"), ".nc")
ncpath <- file.path(root, fname)

# define variables
fillvalue <- 1e32
#var_def_list <- lapply(attributes(my_vpi)$names, function(var) ncvar_def(var, "", time_dim, fillvalue, var))
var_def_list <- list()
for (var in attributes(vpi)$names){
  if (var != "datetime"){
    var_def_list[[var]] <- ncvar_def(var, "", list(lat_dim, lon_dim, time_dim), fillvalue, var)
  } else {
    var_def_list[["solarpos"]] <- ncvar_def("solarpos", "", list(lat_dim, lon_dim, time_dim), fillvalue, "solarpos")
  }
}


ncout <- nc_create(ncpath, var_def_list, force_v4=F)
for (var in names(var_def_list)){
  if (var != "solarpos") {
    #ncout <- nc_create(ncpath, var_def, force_v4=F)
    ncvar_put(ncout, var_def_list[[var]], vpi[[var]])
  } else {
    # add sun elevation angle as additional variable to dataset
    location <- SpatialPoints(data.frame(lon=lon, lat=lat), proj4string = CRS("+proj=longlat +datum=WGS84"))
    data <- solarpos(location, c(vpi$datetime))[,2]
    ncvar_put(ncout, var_def_list[[var]], data)
  }
}

# add global attributes
ncatt_put(ncout, 0, "source", radar)
ncatt_put(ncout, 0, "latitude", lat)
ncatt_put(ncout, 0, "longitude", lon)
ncatt_put(ncout, 0, "fillvalue", fillvalue)
ncatt_put(ncout, 0, "history", paste("F. Lippert", date(), sep=", "))

# close the file, writing data to disk
nc_close(ncout)

print('done')

