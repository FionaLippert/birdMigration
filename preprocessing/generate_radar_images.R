#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

require(bioRad)
require(uvaRadar)
require(lubridate)
require(raster)
require(stars)
require(parallel)
require(MASS)
require(yaml)

root <- args[1]
config = yaml.load_file(file.path(root, "config.yml"))

# set credentials for UvA Radar Data Storage
s3_set_key(username = config$login$username,
           password = config$login$password)

xmin <- config$bbox[[1]] - config$reach
ymin <- config$bbox[[2]] - config$reach
xmax <- config$bbox[[3]] + config$reach
ymax <- config$bbox[[4]] + config$reach
grid <- raster(xmn=xmin, xmx=xmax, ymn=ymin, ymx=ymax, res=config$res)
#img_size <- c(dim(grid)[[2]], dim(grid)[[1]]) # size of final images [pixels]

vertical_integration <- function(timestamp){

    # compute vertically integrated radar composite
    keys <- get_keys(config$radars, timestamp)
    timestamp <- format(timestamp, format="%Y%m%dT%H%M")
    message(timestamp)

    # apply vertical integration to all available radars at time t=ts+dt
    for(k in keys){
      path <- file.path(root, dirname(k))
      if(!dir.exists(path){
        dir.create(path)
        integrate_to_ppi(raster = grid,
                         pvol = retrieve_pvol(vp_key_to_pvol(k)),
                         vp = retrieve_vp(k))
      }
      strs <- st_as_stars(composite$data)
      fname <- paste0(timestamp, ".tif")
      write_stars(strs, file.path(path, fname), driver = "GTiff")
    }
}

vertical_integration(as.POSIXct(args[2], "UTC"))
