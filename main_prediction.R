
source("environment.R")
source("model_UNet.R") 
model <- UNet()
model_name <- "UNet"
tile_size = 15
inputdir<-paste('D:/OneDrive - Hochschule für nachhaltige Entwicklung Eberswalde/[WINMOL]/GIS/predictions/Raster/clipped/')
outputdir<-paste(wd,'output/',sep="")
dataset_name="DS_spruce_2"

img_list <- list.files(inputdir, full.names=TRUE, pattern="\\.tiff$")

"Specify a model path or use the recently trained model"
model_path=".\\models\\SpecDS_UNet_Mask-RCNN_512_All\\UNet_SpecDS_UNet_Mask-RCNN_512_All_model_2023-02-27_211141.hdf5"

if (is.null(model)){
  model<-load_model_hdf5(filepath=model_path, compile=FALSE)
  model_name=strsplit(tail(unlist(strsplit(model_path,split="\\\\")),n=1),split="\\.")[[1]][1]
  model %>% compile(
   optimizer = optimizer,
    loss = F1Score_loss,
    metrics =  list(Precision, Recall, F1Score))
}

#Batch prediction of the orthomosaics stored in input directory
for (img2path in img_list) {
  start.time<-proc.time()
  
  
  img <- brick(img2path)
  file_name=strsplit(tail(unlist(strsplit(img@file@name,split="\\\\")),n=1),split="\\.")[[1]][1]
  img <- dropLayer(img,4)
  px_per_tile= as.integer(floor(tile_size/res(img)[1]))
  xmin=raster::xmin(img)
  ymin=raster::ymin(img)
  ymax=raster::ymax(img)
  res=res(img)[1]
  crs=img@crs@projargs
  
  
  #img<-as.array(as(img,'SpatialGridDataFrame'))
  img <- array(img, dim=c(ncol(img),nrow(img),nlayers(img)))
  
  
  
  img<-img/255
  img<-aperm(img, c(2,1,3))
  
  
  xtiles=floor(length(img[1,,1])/(px_per_tile))+1
  ytiles=floor(length(img[,1,1])/(px_per_tile))+1
  
  img2 <- array(-255,c(ytiles*(px_per_tile)+2,xtiles*(px_per_tile)+2,3))
  img2[0:nrow(img),0:ncol(img),]<-img[,,]
  img<-img2
  IMG_width_=round(((px_per_tile-2)/px_per_tile)*IMG_width/2)*2
  prediction<- array(numeric(),c(ytiles*IMG_width_,xtiles*IMG_width_))
  overlapp=(IMG_width-IMG_width_)/2
  
  #convert to tensor
  
  
  for(i in 1:ytiles) {
    x<-((i-1)*(px_per_tile-2)+1)
    for(j in 1:xtiles){
      y<-((j-1)*(px_per_tile-2)+1)
      tile<-img[x:(x+px_per_tile-1),y:(y+px_per_tile-1),1:3]
      tile<-tensorflow::tf$convert_to_tensor(tile,dtype = tensorflow::tf$float32)  
      tile<- tensorflow::tf$image$resize(tile, size= tfdatasets::shape(IMG_width, IMG_height),method="bicubic", antialias=TRUE)
      tile<-tensorflow::tf$reshape(tile,tfdatasets::shape(1,IMG_width,IMG_width,3))
      pred<-predict_on_batch(model, tile)
   #   predicted_mask =pred[,,,1] %>%as.raster() 
   #   plot(predicted_mask)
      prediction[((i-1)*IMG_width_+1):((i-1)*IMG_width_+IMG_width_),((j-1)*IMG_width_+1):((j-1)*IMG_width_+IMG_width_)] <-pred[,(1+overlapp):(IMG_width-overlapp),(1+overlapp):(IMG_width-overlapp),1]
      }
  }
  data <- raster::raster(prediction)
  data@extent@xmin <- xmin
  data@extent@xmax <- xmin+((xtiles*(px_per_tile-2)+2)*res)
  data@extent@ymin <- ymax-((ytiles*(px_per_tile-2)+2)*res)
  data@extent@ymax <- ymax
  data@crs@projargs <- crs
  
  
  end.time<-proc.time()
  run.time=end.time-start.time
  print(run.time)
  
  
  
    
    
  writeRaster(data,paste(outputdir,"predic_",model_name,"_",dataset_name,"_",file_name,"_",format(Sys.time(),"%Y-%m-%d_%H%M%S"),".tiff",sep=""),"GTiff",overwrite=TRUE)
}


 

