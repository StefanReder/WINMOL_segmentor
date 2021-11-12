


if (is.null(model)){
model<-keras::load_model_hdf5(filepath="C:/temp/output/aug10/UNet_aug10_model_2021-03-29_193609.hdf5", compile=FALSE)
model %>% compile(
  optimizer = optimizer,
  loss = F1Score_loss,
  metrics =  list(Precision, Recall, F1Score))
}

img2path <-"c:/temp/img/pix4dmission1_transparent_mosaic_group1.tif"
img <- brick(img2path)
img <- dropLayer(img,4)

img_predicted<-NULL
###Moving window 20x20m
  x_origin<-xmin(img)+2
  y_origin<-ymin(img)+2

  xtiles=floor((xmax(img)-xmin(img)-4)/16)-1
  ytiles=floor((ymax(img)-ymin(img)-4)/16)-1
 
 # xtiles<-10
 # ytiles<-10

 

  for(i in 1:xtiles) {
    x<-x_origin+((i-1)*16)
  
    
  for(j in 1:ytiles){
    y<-y_origin+((j-1)*16)
    footprint<-raster::raster(xmn=x,xmx=x+16,ymn=y,ymx=y+16,crs=raster::crs(img),resolution=0.05)
    tile_out<-raster::crop(img,raster::extent(extent(footprint@extent@xmin-2,footprint@extent@xmax+2,footprint@extent@ymin-2,footprint@extent@ymax+2)))
    tile<-tile_out/255
    tile<-as.array(tile)#as(tile,'SpatialGridDataFrame'))
   # tile<-aperm(tile, c(2,1,3))
    tile<-tensorflow::tf$convert_to_tensor(tile,dtype = tensorflow::tf$float32)
    tile<- tensorflow::tf$image$resize(tile, size= tfdatasets::shape(IMG_width, IMG_height),method="nearest")
    tile<-tensorflow::tf$reshape(tile,tfdatasets::shape(1,IMG_width,IMG_width,3))

   
   prediction <-stats::predict(model, tile)

   data <- raster::raster(prediction[,,,1])
   data@extent@xmin <- raster::xmin(tile_out)
   data@extent@xmax <- raster::xmax(tile_out)
   data@extent@ymin <- raster::ymin(tile_out)
   data@extent@ymax <- raster::ymax(tile_out)
   data@crs@projargs <- footprint@crs@projargs
   
   
   data<-raster::crop(data,extent(footprint))
   data<-resample(data,footprint)
  # extent_new=extent(data)+c(16*res(data)[1],-16*res(data)[1],16*res(data)[1],-16*res(data)[1])
  # data<-crop(data,extent_new)
   

  if(is.null(img_predicted))
   {
     img_predicted<-data
     }  else{
   img_predicted<-mosaic(data,fun=mean,img_predicted)
  }
   
  plot(img_predicted)
  } 
}

  
  writeRaster(img_predicted ,paste(outputdir,"predic_",model_name,"_",dataset_name,"_",format(Sys.time(),"%Y-%m-%d_%H%M%S"),".tiff",sep=""),"GTiff",overwrite=TRUE)
  #   predicted_mask =prediction[,,,1] %>%
  #        as.raster() %>% magick::image_read()
  
  
  #   tile_out<-as.imlist(tile_out) %>% imappend("c")
  #   tile_out<-image_read(tile_out)
  #   out <- magick::image_append(c(
  #     magick::image_append(tile_out, stack = TRUE),
  #      magick::image_append(predicted_mask, stack = TRUE)))
  
  #   plot(out)