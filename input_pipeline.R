
###Data Augmentation

imgAug <- function(img, rndint) {img %>%
    tf$image$random_flip_up_down(seed=rndint) %>%
    tf$image$random_flip_left_right(seed=rndint)}

maskAug <- function(mask, rndint) {mask %>%
    tf$image$random_flip_up_down(seed=rndint) %>%
    tf$image$random_flip_left_right(seed=rndint)}

#Create function to change brightness, contrast, saturation, and hue randomly
random_bcsh <- function(img) {img %>% 
    tf$image$random_brightness(max_delta = 0.1) %>% 
    tf$image$random_contrast(lower = 0.95, upper = 1.05) %>% 
    tf$image$random_saturation(lower = 0.95, upper = 1.05) %>% 
    tf$image$random_hue(max_delta = 0.1) %>%
    # make sure we still are between 0 and 1
    tf$clip_by_value(0, 1)}

####Load data set with pre-processing chain
create_dataset <- function(data, train, batch_size =4L) {
  dataset <- data %>% 
    tensor_slices_dataset() %>% 
    dataset_map(~.x %>% list_modify(
      img = tf$image$decode_jpeg(tf$io$read_file(.x$img), channels = 3)%>%
        tf$image$convert_image_dtype(dtype = tf$float32)%>%
        tf$image$resize(size = shape(IMG_width, IMG_height),method="nearest"),
      mask = tf$image$decode_gif(tf$io$read_file(.x$mask))[1,,,][,,1,drop=FALSE] %>%
        tf$image$convert_image_dtype(dtype = tf$float32)%>%
        tf$image$resize(size = shape(IMG_width, IMG_height), method="nearest")))
  # data augmentation performed on training set only
  if (train) {
    rndint <- sample(1:100, 1)
    dataset <- dataset %>% 
      dataset_map(~.x %>% list_modify(
        img = imgAug(.x$img, rndint=rndint)%>%random_bcsh(),
        mask = maskAug(.x$mask, rndint=rndint)))%>%
      dataset_shuffle(buffer_size = batch_size*IMG_width)
      }
  
  dataset <- dataset %>%
    dataset_batch(batch_size)%>%
    dataset_map(unname)%>%
    dataset_prefetch(buffer_size = tf$data$experimental$AUTOTUNE)
}
