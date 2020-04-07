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
#grid <- raster(xmn=lat_min, xmx=lat_max, ymn=lon_min, ymx=lon_max, res=config$res)
#img_size <- c(dim(grid)[[2]], dim(grid)[[1]]) # size of final images [pixels]

vertical_integration <- function(timestamp){

    # compute vertically integrated radar composite
    keys <- get_keys(config$radars, timestamp)
    timestamp <- format(timestamp, format="%Y%m%dT%H%M")
    #message(timestamp)

    # apply vertical integration to all available radars at time t=ts+dt
    for(k in keys){
      key_split <- strsplit(k, .Platform$file.sep)[[1]]
      country <- key_split[[1]]
      rname <- key_split[[2]]
      fname <- paste0(country, rname, "_", timestamp, ".h5")
      output_path <- file.path(root, country, rname, fname)
      message(output_path)

      h5createFile(output_path)
      h5createGroup(output_path, "data")
      #path <- file.path(root, dirname(k))
      #if(!dir.exists(path)){
      #dir.create(path, recursive=TRUE)
      pvol = retrieve_pvol(vp_key_to_pvol(k))
      vp = retrieve_vp(k)
      ppi <- integrate_to_ppi(pvol = pvol, vp = vp,
                              res = config$res,
                              xlim = c(lon_min,lon_max),
                              ylim = c(lat_min,lat_max))
      r <- raster(ppi$data)
      r_attr = attributes(r)
      quantities = names(ppi$data)

      for (q in quantities){
        data = as(ppi$data[q], "matrix")
        attr(data, "quantity") <- q
        h5write(data, output_path, paste0("data/", q))
      }

      fid = H5Fopen(output_path)
      h5g = H5Gopen(fid, "attributes")

      h5writeAttribute(attr = lon_min, h5obj = h5g, name = "lon_min")
      h5writeAttribute(attr = lat_min, h5obj = h5g, name = "lat_min")
      h5writeAttribute(attr = lon_max, h5obj = h5g, name = "lon_max")
      h5writeAttribute(attr = lat_max, h5obj = h5g, name = "lat_max")
      h5writeAttribute(attr = r_attr$ncols, h5obj = h5g, name = "ncols")
      h5writeAttribute(attr = r_attr$nrows, h5obj = h5g, name = "nrows")
      h5writeAttribute(attr = res(r), h5obj = h5g, name = "resolution")
      h5writeAttribute(attr = ppi$datetime, h5obj = h5g, name = "time")
      h5writeAttribute(attr = vp$attributes$where$lat, h5obj = h5g, name = "lat_radar")
      h5writeAttribute(attr = vp$attributes$where$lon, h5obj = h5g, name = "lon_radar")
      H5Gclose(h5g)
      H5Fclose(fid)
      h5closeAll()

      #strs <- st_as_stars(result$data)
      #message(length(result$data[[config$quantity]]))
      #fname <- paste0(config$quantity, "_", timestamp, ".tif")
      #write_stars(strs[config$quantity], file.path(path, fname), driver = "GTiff")
    }
}

vertical_integration(as.POSIXct(args[2], "UTC"))
