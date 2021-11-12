

cost_eval <- function(model,stage){

if(stage==1){
  eval_set<-val_set
}  
  
if(stage==2){eval_data <- tibble(
  img=list.files(paste(tempdir,"TestDS_2/train/",sep=""), full.names=TRUE, pattern="\\.jpeg$"),
  mask=list.files(paste(tempdir,"TestDS_2/mask/",sep=""), full.names=TRUE, pattern="\\.gif$"))
  eval_set <- create_dataset(eval_data, train=FALSE)

}   
  




score <- model %>% evaluate(eval_set,  batch_size = 4)

batch <- eval_set %>% as_iterator() %>% iter_next()
predictions <- predict(model, batch[[1]])

images <- tibble(
  image = batch[[1]] %>% array_branch(1),
  predicted_mask =predictions[,,,1] %>%array_branch(1), 
  mask = batch[[2]][,,,1] %>% array_branch(1))%>% 
  #   sample_n(4) %>% 
  map_depth(2, function(x) {
    as.raster(x) %>% magick::image_read()
  }) %>% 
  map(~do.call(c, .x))


out <- magick::image_append(c(
  magick::image_append(images$mask, stack = TRUE),
  magick::image_append(images$image, stack = TRUE), 
  magick::image_append(images$predicted_mask, stack = TRUE)))

plot(out)

gc()
}
