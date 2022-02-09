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

# set credentials for UvA Radar Data Storage
s3_set_key(username = config$login$username,
           password = config$login$password)

#lon_min <- config$bounds[[1]] - config$reach
#lat_min <- config$bounds[[2]] - config$reach
#lon_max <- config$bounds[[3]] + config$reach
#lat_max <- config$bounds[[4]] + config$reach
grid <- raster(xmn=config$lon_min, xmx=config$lon_max, ymn=config$lat_min, ymx=config$lat_max, res=config$res)

img_size <- c(dim(grid)[[2]], dim(grid)[[1]]) # size of final images [pixels]

vertical_integration <- function(datetime){

    # compute vertically integrated radar composite
    keys <- get_keys(config$radars, datetime)
    timestamp <- format(datetime, format="%Y%m%dT%H%M")
    #message(timestamp)

    # apply vertical integration to all available radars at time t=ts+dt
    ppi_list <- list()
    for(k in keys){
      #key_split <- strsplit(k, .Platform$file.sep)[[1]]
      #country <- key_split[[1]]
      #rname <- key_split[[2]]
      #fname <- paste0(country, rname, "_", timestamp, ".h5")
      #output_path <- file.path(root, country, rname, fname)
      #message(output_path)

      #h5createFile(output_path)
      #groupname <- paste0(config$quantity, "_data")
      #h5createGroup(output_path, groupname)

      pvol = retrieve_pvol(vp_key_to_pvol(k), param="all")
      vp = retrieve_vp(k)
      print(paste('//////////', is.finite(config$filter$DR_min)))
      if(is.finite(config$filter$DR_min)){
        pvol <-calculate_parameter(pvol,
                       ZDR=DBZH-DBZV,
                       ZDRL=10^(ZDR/10),
                       DRL=((ZDRL+1-2*sqrt(ZDRL)*RHOHV)/(ZDRL+1+2*sqrt(ZDRL)*RHOHV)),
                       DPR=10*log10(DRL))

        pvol <- calculate_param(pvol, DBZH = ifelse(DPR > config$filter$DR_min & DBZH < config$filter$DBZ_max, DBZH, NaN))

        #pvol <- calculate_param(pvol, DR = 10 * log10((ZDR + 1 - 2 * ZDR^0.5 * RHOHV) /
        #                                (ZDR + 1 + 2 * ZDR^0.5 * RHOHV)))
        #pvol <- calculate_param(pvol, DBZH = ifelse(DR > config$filter$DR_min & DBZH < config$filter$DBZ_max, DBZH, NaN))
      }
      ppi <- integrate_to_ppi(pvol = pvol, vp = vp, raster = grid,
                              #res = config$res,
                              #xlim = c(lon_min,lon_max),
                              #ylim = c(lat_min,lat_max),
                              param = "DBZH",
                              param_ppi = config$quantity)
      
      if(!all(is.na(ppi$data[[config$quantity]]))){
        ppi_list[[k]] <- ppi
      }
      #r <- raster(ppi$data)
      #r_attr = attributes(r)

      #data = as(ppi$data[config$quantity], "matrix")
      #h5write(data, output_path, paste0(groupname, "/data"))


      #h5createGroup(output_path, "where")
      #h5createGroup(output_path, "how")
      #h5createGroup(output_path, "what")

      #fid = H5Fopen(output_path)

      #h5g = H5Gopen(fid, "where")
      #h5writeAttribute(attr = r_attr$crs, h5obj = h5g, name = "projdef")
      #h5writeAttribute(attr = vp$attributes$where$lat, h5obj = h5g, name = "lat")
      #h5writeAttribute(attr = vp$attributes$where$lon, h5obj = h5g, name = "lon")
      #H5Gclose(h5g)

      #h5g = H5Gopen(fid, "how")
      #h5writeAttribute(attr = ppi$geo$bbox[2], h5obj = h5g, name = "lon_min")
      #h5writeAttribute(attr = ppi$geo$bbox[1], h5obj = h5g, name = "lat_min")
      #h5writeAttribute(attr = ppi$geo$bbox[4], h5obj = h5g, name = "lon_max")
      #h5writeAttribute(attr = ppi$geo$bbox[3], h5obj = h5g, name = "lat_max")
      #h5writeAttribute(attr = res(r), h5obj = h5g, name = "xscale")
      #h5writeAttribute(attr = res(r), h5obj = h5g, name = "yscale")
      #h5writeAttribute(attr = r_attr$ncols, h5obj = h5g, name = "xsize")
      #h5writeAttribute(attr = r_attr$nrows, h5obj = h5g, name = "ysize")

      #h5g = H5Gopen(fid, "what")
      #h5writeAttribute(attr = config$quantity, h5obj = h5g, name = "quantity")
      #h5writeAttribute(attr = "IMAGE", h5obj = h5g, name = "object")
      #h5writeAttribute(attr = paste0(country, "/", rname), h5obj = h5g, name = "source")
      #h5writeAttribute(attr = ppi$datetime, h5obj = h5g, name = "datetime")
      #H5Fclose(fid)
      #h5closeAll()
    }
    if (length(ppi_list) > 0) {
      composite <- composite_ppi(ppi_list, param=config$quantity, dim=img_size)
      r <- raster(composite$data)
      r_attr <- attributes(r)
      r_res <- res(r)
      #r[] <- 1:ncell(r)
      #data_matrix <- as.matrix(r)
     
      #print(paste('//////////', ncol(r), nrow(r)))
      #print(paste('----------', ncol(grid), nrow(grid)))
      #print(paste('----------', dim(as(composite$data[config$quantity], 'matrix'))))

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
      #ncatt_put(ncout, "lon", "longitude", "axis", "X")
      #ncatt_put(ncout, "lat", "latitude", "axis", "Y")
      #ncatt_put(ncout, "time", "time", "axis", "T")

      # add global attributes
      ncatt_put(ncout, 0, "projdef", as.character(r_attr$crs))
      ncatt_put(ncout, 0, "resolution", r_res)
      ncatt_put(ncout, 0, "source", config$radars)
      ncatt_put(ncout, 0, "fillvalue", fillvalue)
      #ncatt_put(ncout, 0, "datetime", as.character(datetime))
      ncatt_put(ncout, 0, "history", paste("F. Lippert", date(), sep=", "))
      print(paste('----------------', date()))

      # close the file, writing data to disk
      nc_close(ncout)
      #print('//////////////////// data written to disk ///////////////')
    }


    # fname <- paste0("composite_", timestamp, ".h5")
    # output_path <- file.path(root, fname)
    # message(output_path)
    # h5createFile(output_path)
    #
    # groupname <- paste0(config$quantity, "_data")
    # h5createGroup(output_path, groupname)
    # data = as(composite$data, "matrix")
    # h5write(data, output_path, paste0(groupname, "/data"))

    # h5createGroup(output_path, "where")
    # h5createGroup(output_path, "how")
    # h5createGroup(output_path, "what")
    #
    # fid = H5Fopen(output_path)
    #
    # h5g = H5Gopen(fid, "where")
    # h5writeAttribute(attr = as.character(r_attr$crs), h5obj = h5g, name = "projdef")
    # H5Gclose(h5g)
    #
    # h5g = H5Gopen(fid, "how")
    # h5writeAttribute(attr = lon_min, h5obj = h5g, name = "lon_min")
    # h5writeAttribute(attr = lat_min, h5obj = h5g, name = "lat_min")
    # h5writeAttribute(attr = lon_max, h5obj = h5g, name = "lon_max")
    # h5writeAttribute(attr = lat_max, h5obj = h5g, name = "lat_max")
    # h5writeAttribute(attr = res(r), h5obj = h5g, name = "resolution")
    # h5writeAttribute(attr = r_attr$ncols, h5obj = h5g, name = "xsize")
    # h5writeAttribute(attr = r_attr$nrows, h5obj = h5g, name = "ysize")
    #
    # h5g = H5Gopen(fid, "what")
    # h5writeAttribute(attr = config$quantity, h5obj = h5g, name = "quantity")
    # h5writeAttribute(attr = "IMAGE", h5obj = h5g, name = "object")
    # h5writeAttribute(attr = config$radars, h5obj = h5g, name = "source")
    # h5writeAttribute(attr = as.character(datetime), h5obj = h5g, name = "datetime")
    # H5Fclose(fid)
    #
    # h5closeAll()
}

vertical_integration(as.POSIXct(args[2], "UTC"))
