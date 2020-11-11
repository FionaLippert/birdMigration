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
print(radar)
#root <- "~/birdMigration/preprocessing"
config = yaml.load_file(file.path(root, "config.yml"))

# set credentials for UvA Radar Data Storage
s3_set_key(username = config$login$username,
           password = config$login$password)

print(radar)

my_vpts <- get_vpts(radars = radar,
                    time = seq(from = as.POSIXct(config$ts, tz = "UTC"),
                               to = as.POSIXct(config$te, tz = "UTC"),
                               by = paste(config$tr, "mins")),
                    #with_db=T (not working for some reasaon. something wrong with keyring?)
                    )
# make a subselection for night time only
#index_night <- check_night(my_vpts)
#my_vpts_night <- my_vpts[index_night]

my_vpts <- regularize_vpts(my_vpts)
my_vpi <- integrate_profile(my_vpts)

#plot(my_vpi, night_shade = FALSE, quantity="vid")

#define dimensions
time_dim <- ncdim_def("time", "seconds since 1970-01-01 00:00:00", as.numeric(c(my_vpi$datetime)), unlim=FALSE)


# create netcdf file
fname <- paste0("vpi_", str_remove(radar, "/"), ".nc")
ncpath <- file.path(root, fname)

# define variables
fillvalue <- 1e32
#var_def_list <- lapply(attributes(my_vpi)$names, function(var) ncvar_def(var, "", time_dim, fillvalue, var))
var_def_list <- list()
for (var in attributes(my_vpi)$names){
  if (var != "datetime"){
    var_def_list[[var]] <- ncvar_def(var, "", time_dim, fillvalue, var)
  } else {
    var_def_list[["solarpos"]] <- ncvar_def("solarpos", "", time_dim, fillvalue, "solarpos")
  }
}

# extract radar location from vpts object
lat <- my_vpts$attributes$where$lat
lon <- my_vpts$attributes$where$lon

ncout <- nc_create(ncpath, var_def_list, force_v4=F)
for (var in names(var_def_list)){
  if (var != "solarpos") {
    #ncout <- nc_create(ncpath, var_def, force_v4=F)
    ncvar_put(ncout, var_def_list[[var]], my_vpi[[var]])
  } else {
    # add sun elevation angle as additional variable to dataset
    location <- SpatialPoints(data.frame(lon=lon, lat=lat), proj4string = CRS("+proj=longlat +datum=WGS84"))
    data <- solarpos(location, c(my_vpi$datetime))[,2]
    ncvar_put(ncout, var_def_list[[var]], data)
  }
}
print(location)
# add global attributes
ncatt_put(ncout, 0, "source", radar)
ncatt_put(ncout, 0, "lat", lat)
ncatt_put(ncout, 0, "lon", lon)
ncatt_put(ncout, 0, "fillvalue", fillvalue)
ncatt_put(ncout, 0, "history", paste("F. Lippert", date(), sep=", "))

# close the file, writing data to disk
nc_close(ncout)

print('done')

