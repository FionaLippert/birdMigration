#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

require(bioRad)
require(uvaRadar)
require(lubridate)
require(raster)
require(stars)
require(parallel)
require(MASS)
#require(numform)
require(yaml)

#num_cores <- detectCores() - 1
subdir <- args[1]
config = yaml.load_file(file.path(subdir, "config.yml"))

# set credentials for UvA Radar Data Storage
s3_set_key(username = config$login$username,
           password = config$login$password)

ts <- as.POSIXct(config$ts, "UTC")  # POSIXct start time
te <- as.POSIXct(config$te, "UTC")  # POSIXct end time
tl   <- as.numeric(difftime(te, ts, units = "mins"))    # total length in minutes
tseq <- seq(0, tl, config$tr)                                  # delta t sequence

#bbox <- st_bbox(get_radars_df(config$radars)$geometry)
#extent_x <- bbox$xmax - bbox$xmin
#extent_y <- bbox$ymax - bbox$ymin

xmin <- config$bbox[[1]] - config$reach
ymin <- config$bbox[[2]] - config$reach
xmax <- config$bbox[[3]] + config$reach
ymax <- config$bbox[[4]] + config$reach
grid <- raster(xmn=xmin, xmx=xmax, ymn=ymin, ymx=ymax, res=config$res)
img_size <- c(dim(grid)[[2]], dim(grid)[[1]]) # size of final images [pixels]


#subdir <- file.path(config$data$tiff, paste(ts, "-", te))

#log <- file(file.path(subdir, "log.txt"), open="w")
#sink(log, type='message', append=TRUE)

message(paste(length(tseq), 'frames to be processed:'))


create_composite <- function(timestamp){

    #timestamp = ts + minutes(tseq[[idx]])
    # compute vertically integrated radar composite
    keys <- get_keys(config$radars, timestamp)

    timestamp <- as.character(timestamp)
    if(nchar(timestamp) < 11){
      timestamp <- paste(timestamp, '00:00:00')
    }
    message(timestamp)

    # apply vertical integration to all available radars at time t=ts+dt
    # and combine the resulting ppi's into composite raster

    # TODO: load pvol and vp prior to parallel execution of vertical integration
    if(length(keys) > 0) {
      names(keys) <- keys
      pvol_list <- sapply(keys, function(k) { retrieve_pvol(vp_key_to_pvol(k))},
                    simplify = FALSE, USE.NAMES = TRUE)

      vp_list <- sapply(keys, retrieve_vp, simplify = FALSE, USE.NAMES = TRUE)

      # TODO; fix error arising when vp$data[['eta']] is all NAN (with BE/WID at 2016-10-01 00:15)
      # maybe fix it by using 'dens' instead of 'eta'?
      ppi_list <- lapply(keys, function(k) {
                integrate_to_ppi(pvol=pvol_list[[k]],
                                 vp=vp_list[[k]],
                                 raster=grid)
                })#, mc.cores=num_cores-1)

      # TODO: adjust when new bioRad release is available (with res as input)
      composite <- composite_ppi(ppi_list, param=config$quantity, dim=img_size)

      fname <- file.path(subdir, paste0(timestamp, ".tif"))
      strs <- st_as_stars(composite$data)
      write_stars(strs, fname, driver = "GTiff")

    }
    else{
      message(paste('no data for timestamp', timestamp))
    }
}

#jobs <- seq(length(tseq))
#system.time({
#  results <- mclapply(jobs, create_composite, mc.cores = num_cores)
#})
#message(paste(Reduce("+", results), 'of', length(jobs), 'frames have been processed successfully'))

create_composite(as.POSIXct(args[2], "UTC"))

#sink(NULL,type='message')
