#!/usr/bin/env Rscript
# Title     : VPTS2VP
# Objective : load vertical profile time series from disk, compute average vertical profile, and save as netcdf
# Created by: fiona

args = commandArgs(trailingOnly=TRUE)

require(lubridate)
require(bioRad)
require(yaml)
require(stringr)
require(ncdf4)

root <- args[1]
filepath <- args[2]

config = yaml.load_file(file.path(root, "config.yml"))

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

filter_dbzh <- function(vpts, threshold=7,height=1000, agl_max=Inf, drop=F, quantity="DBZH"){
  height_index_max <- ((vpts$attributes$where$height + agl_max) %/% vpts$attributes$where$interval)
  height_index_max <- min(dim(vpts)[2],height_index_max)
  height_range <- colSums(vpts$data[[quantity]][1:height_index_max,]>threshold,na.rm=T)*vpts$attributes$where$interval
  index <- which(height_range > height)
  if(length(index)==0) return(vpts)
  # if remove, drop the profiles
  if(drop) return(vpts[-index])
  # otherwise set the density field to NA, but keep the profile
  vpts$data$DBZH[,index] <- NA
  vpts$data$dens[,index] <- NA
  vpts
}

sd_vvp_threshold(vpts) <- 0

# make a subselection for night time only
index_night <- check_night(vpts)
vpts <- vpts[index_night]

# regularize to hourly resolution
#vpts <- regularize_vpts(vpts, date_min=begin, date_max=end, interval=1, fill=7.5, units="hours")


# remove some more rain
vpts <- filter_dbzh(vpts, threshold=500, height=2000, agl_max=Inf, drop=F, quantity="dens")


############################################################
# TODO: compute average vertical profile for this radar
############################################################
# print(attributes(vpts$data)$names)

#define dimensions
height_dim <- ncdim_def("height", "meters",
                      as.integer(c(vpts$height)), unlim=FALSE)
#time_dim <- ncdim_def("time", "seconds since 1970-01-01 00:00:00",
#                      as.integer(c(vpts$datetime)), unlim=FALSE)
lat_dim <- ncdim_def("lat", "degrees north", as.double(c(lat)), unlim=FALSE)
lon_dim <- ncdim_def("lon", "degrees south", as.double(c(lon)), unlim=FALSE)

# create netcdf file
fname <- paste0("vp_avg_", radar_odim_format, ".nc")
ncpath <- file.path(root, fname)

# define variables
fillvalue <- 1e32
var_def_list <- list()
for (var in attributes(vpts$data)$names){
  # var_def_list[[var]] <- ncvar_def(var, "", list(lat_dim, lon_dim, height_dim, time_dim), fillvalue, var)
  var_def_list[[var]] <- ncvar_def(var, "", list(lat_dim, lon_dim, height_dim), fillvalue, var)
}
ncout <- nc_create(ncpath, var_def_list, force_v4=F)
for (var in names(var_def_list)){
  if (var %in% c("dens", "heading", "dd")){
    # print(dim(vpts$data[[var]]))
    # print(dim(apply(vpts$data[[var]], 1, mean)))
    # ncvar_put(ncout, var_def_list[[var]], vpts$data[[var]])
    vpts_var <- vpts$data[[var]]
    vpts_var[is.nan(vpts_var)] <- 0
    avg_vp <- apply(vpts_var, 1, mean) #function(x) mean(na.omit(x)))
    ncvar_put(ncout, var_def_list[[var]], avg_vp)
  }
}

# add global attributes
ncatt_put(ncout, 0, "source", radar_odim_format)
ncatt_put(ncout, 0, "latitude", lat)
ncatt_put(ncout, 0, "longitude", lon)
ncatt_put(ncout, 0, "altitude", radar_alt)
ncatt_put(ncout, 0, "fillvalue", fillvalue)
ncatt_put(ncout, 0, "history", paste("F. Lippert", date(), sep=", "))

# close the file, writing data to disk
nc_close(ncout)
print('done')
