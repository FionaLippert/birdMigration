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



root <- args[1]
config = yaml.load_file(file.path(root, "config.yml"))

# set credentials for UvA Radar Data Storage
s3_set_key(username = config$login$username,
           password = config$login$password)

lat_min <- config$bounds[[1]] - config$reach
lon_min <- config$bounds[[2]] - config$reach
lat_max <- config$bounds[[3]] + config$reach
lon_max <- config$bounds[[4]] + config$reach
grid <- raster(xmn=lat_min, xmx=lat_max, ymn=lon_min, ymx=lon_max, res=0.01)
img_size <- c(dim(grid)[[2]], dim(grid)[[1]]) # size of final images [pixels]

vertical_integration <- function(timestamp){

    # compute vertically integrated radar composite
    keys <- get_keys(config$radars, timestamp)
    timestamp <- format(timestamp, format="%Y%m%dT%H%M")
    #message(timestamp)

    # apply vertical integration to all available radars at time t=ts+dt
    ppi_list <- list()
    for(k in keys){
      key_split <- strsplit(k, .Platform$file.sep)[[1]]
      country <- key_split[[1]]
      rname <- key_split[[2]]
      fname <- paste0(country, rname, "_", timestamp, ".h5")
      output_path <- file.path(root, country, rname, fname)
      message(output_path)

      h5createFile(output_path)
      groupname <- paste0(config$quantity, "_data")
      h5createGroup(output_path, groupname)

      pvol = retrieve_pvol(vp_key_to_pvol(k))
      vp = retrieve_vp(k)
      ppi <- integrate_to_ppi(pvol = pvol, vp = vp, raster = grid,
                              #res = config$res,
                              #xlim = c(lon_min,lon_max),
                              #ylim = c(lat_min,lat_max),
                              param_ppi = config$quantity)
      ppi_list[[k]] <- ppi
      r <- raster(ppi$data)
      r_attr = attributes(r)
      #quantities = names(ppi$data)

      #for (q in quantities){
      data = as(ppi$data[config$quantity], "matrix")
      #subgroup <- paste0(groupname, "/", q)
      #h5createGroup(output_path, subgroup)
      h5write(data, output_path, paste0(groupname, "/data"))
      #h5createGroup(output_path, paste0(groupname, "/what"))

      #fid = H5Fopen(output_path)
      #h5g = H5Gopen(fid, paste0(subgroup, "/what"))
      #h5writeAttribute(attr = q, h5obj = h5g, name = "quantity")
      #H5Gclose(h5g)
      #H5Fclose(fid)
      #}

      h5createGroup(output_path, "where")
      h5createGroup(output_path, "how")
      h5createGroup(output_path, "what")

      fid = H5Fopen(output_path)

      h5g = H5Gopen(fid, "where")
      h5writeAttribute(attr = r_attr$crs, h5obj = h5g, name = "projdef")
      h5writeAttribute(attr = vp$attributes$where$lat, h5obj = h5g, name = "lat")
      h5writeAttribute(attr = vp$attributes$where$lon, h5obj = h5g, name = "lon")
      H5Gclose(h5g)

      h5g = H5Gopen(fid, "how")
      h5writeAttribute(attr = ppi$geo$bbox[2], h5obj = h5g, name = "lon_min")
      h5writeAttribute(attr = ppi$geo$bbox[1], h5obj = h5g, name = "lat_min")
      h5writeAttribute(attr = ppi$geo$bbox[4], h5obj = h5g, name = "lon_max")
      h5writeAttribute(attr = ppi$geo$bbox[3], h5obj = h5g, name = "lat_max")
      h5writeAttribute(attr = res(r), h5obj = h5g, name = "xscale")
      h5writeAttribute(attr = res(r), h5obj = h5g, name = "yscale")
      h5writeAttribute(attr = r_attr$ncols, h5obj = h5g, name = "xsize")
      h5writeAttribute(attr = r_attr$nrows, h5obj = h5g, name = "ysize")

      h5g = H5Gopen(fid, "what")
      h5writeAttribute(attr = config$quantity, h5obj = h5g, name = "quantity")
      h5writeAttribute(attr = "IMAGE", h5obj = h5g, name = "object")
      h5writeAttribute(attr = paste0(country, "/", rname), h5obj = h5g, name = "source")
      #h5writeAttribute(attr = format(ppi$datetime, format="%Y%m%d"), h5obj = h5g, name = "date")
      #h5writeAttribute(attr = format(ppi$datetime, format="%H%M%S"), h5obj = h5g, name = "time")
      h5writeAttribute(attr = ppi$datetime, h5obj = h5g, name = "datetime")
      H5Fclose(fid)
      h5closeAll()
    }
    composite <- composite_ppi(ppi_list, param=config$quantity, dim=img_size)

    fname <- paste0("composite_", timestamp, ".h5")
    output_path <- file.path(root, fname)
    h5createFile(output_path)
    groupname <- paste0(config$quantity, "_data")
    h5createGroup(output_path, groupname)
    data = as(composite$data, "matrix")
    h5write(data, output_path, paste0(groupname, "/data"))
    h5closeAll()
}

vertical_integration(as.POSIXct(args[2], "UTC"))
