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
require(rhdf5)
require(stringr)
require(ncdf4)



root <- args[1]
config = yaml.load_file(file.path(root, "config.yml"))

grid <- raster(xmn=config$lon_min, xmx=config$lon_max, ymn=config$lat_min, ymx=config$lat_max, res=config$res)

img_size <- c(dim(grid)[[2]], dim(grid)[[1]]) # size of final images [pixels]

vertical_integration <- function(datetime){

    # compute vertically integrated radar composite
    keys <- get_keys(config$radars, datetime)
    timestamp <- format(datetime, format="%Y%m%dT%H%M")

    # apply vertical integration to all available radars at time t=ts+dt
    ppi_list <- list()
    for(k in keys){

      pvol = retrieve_pvol(vp_key_to_pvol(k), param="all")
      vp = retrieve_vp(k)

      if(is.finite(config$filter$DR_min)){
        pvol <-calculate_parameter(pvol,
                       ZDR=DBZH-DBZV,
                       ZDRL=10^(ZDR/10),
                       DRL=((ZDRL+1-2*sqrt(ZDRL)*RHOHV)/(ZDRL+1+2*sqrt(ZDRL)*RHOHV)),
                       DPR=10*log10(DRL))

        pvol <- calculate_param(pvol, DBZH = ifelse(DPR > config$filter$DR_min & DBZH < config$filter$DBZ_max, DBZH, NaN))
      }

      ppi <- integrate_to_ppi(pvol = pvol, vp = vp, raster = grid,
                              param = "DBZH",
                              param_ppi = config$quantity)

      if(!all(is.na(ppi$data[[config$quantity]]))){
        ppi_list[[k]] <- ppi
      }
    }
    if (length(ppi_list) > 0) {
      composite <- composite_ppi(ppi_list, param=config$quantity, dim=img_size)
      r <- raster(composite$data)
      r_attr <- attributes(r)
      r_res <- res(r)

      # define dimensions
      lons <- sort(unique(xyFromCell(grid,1:ncell(grid))[,'x']))
      lats <- sort(unique(xyFromCell(grid,1:ncell(grid))[,'y']))
      lon_dim <- ncdim_def("lon", "degrees_east", as.double(lons))
      lat_dim <- ncdim_def("lat", "degrees_north", as.double(lats))
      time_dim <- ncdim_def("time", "seconds since 1970-01-01 00:00:00", as.numeric(c(datetime)), unlim=FALSE)


      # define variable
      fillvalue <- 1e32
      var_def <- ncvar_def(config$quantity, "", list(lon_dim,lat_dim, time_dim), fillvalue, config$quantity)

      print(paste('---------', min(composite$data[[config$quantity]]), max(composite$data[[config$quantity]])))

      # create netcdf file
      fname <- paste0("composite_", timestamp, ".nc")
      ncpath <- file.path(root, fname)
      ncout <- nc_create(ncpath, var_def, force_v4=F)
      ncvar_put(ncout, var_def, as.matrix(composite$data))

      # add global attributes
      ncatt_put(ncout, 0, "projdef", as.character(r_attr$crs))
      ncatt_put(ncout, 0, "resolution", r_res)
      ncatt_put(ncout, 0, "source", config$radars)
      ncatt_put(ncout, 0, "fillvalue", fillvalue)
      ncatt_put(ncout, 0, "history", paste("F. Lippert", date(), sep=", "))
      print(paste('----------------', date()))

      # close the file, writing data to disk
      nc_close(ncout)

    }
}

vertical_integration(as.POSIXct(args[2], "UTC"))
