

UNet <- function(
  input_shape=c(IMG_width,IMG_height,n_Channels), classes=num_classes){
  
  FLAGS <- flags(
    flag_numeric('dropout', 0.1, 'First dropout')
  )

  
  # Input layer
  inputs<-layer_input(shape=input_shape)
  
  # Contracting layers
  
  #C1
  c1<-inputs %>%
    layer_conv_2d(filters = 64,kernel_size = c(3,3), activation = 'relu',
                  kernel_initializer = 'he_normal',padding = 'same', data_format="channels_last", use_bias = FALSE) %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = FLAGS$dropout,seed = 1) %>%
    layer_conv_2d(filters = 64,kernel_size = c(3,3), activation = 'relu',
                  kernel_initializer = 'he_normal',padding = 'same', data_format="channels_last", use_bias = FALSE) %>%
    layer_batch_normalization() 
  
  #C2
  c2<-c1 %>%layer_max_pooling_2d(pool_size =c(2,2)) %>%
    layer_conv_2d(filters = 128,kernel_size = c(3,3), activation = 'relu',
                  kernel_initializer = 'he_normal',padding = 'same', data_format="channels_last", use_bias = FALSE) %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = FLAGS$dropout,seed = 1) %>%
    layer_conv_2d(filters = 128,kernel_size = c(3,3), activation = 'relu',
                  kernel_initializer = 'he_normal',padding = 'same', data_format="channels_last", use_bias = FALSE) %>%
    layer_batch_normalization() 
  
  #C3
  c3<-c2 %>%layer_max_pooling_2d(pool_size =c(2,2))%>%
    layer_conv_2d(filters = 256,kernel_size = c(3,3), activation = 'relu',
                  kernel_initializer = 'he_normal',padding = 'same', data_format="channels_last", use_bias = FALSE) %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = FLAGS$dropout,seed = 1) %>%
    layer_conv_2d(filters = 256,kernel_size = c(3,3), activation = 'relu',
                  kernel_initializer = 'he_normal',padding = 'same', data_format="channels_last", use_bias = FALSE) %>%
    layer_batch_normalization() 
  
  #C4
  c4<-c3 %>%layer_max_pooling_2d(pool_size =c(2,2))%>%
    layer_conv_2d(filters = 512,kernel_size = c(3,3), activation = 'relu',
                  kernel_initializer = 'he_normal',padding = 'same', data_format="channels_last", use_bias = FALSE) %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = FLAGS$dropout,seed = 1) %>%
    layer_conv_2d(filters = 512,kernel_size = c(3,3), activation = 'relu',
                  kernel_initializer = 'he_normal',padding = 'same', data_format="channels_last", use_bias = FALSE) %>%
    layer_batch_normalization() 
  
  #C5
  c5<-c4 %>%layer_max_pooling_2d(pool_size =c(2,2))%>%
    layer_conv_2d(filters = 1024,kernel_size = c(3,3), activation = 'relu',
                  kernel_initializer = 'he_normal',padding = 'same', data_format="channels_last", use_bias = FALSE) %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = FLAGS$dropout,seed = 1) %>%
    layer_conv_2d(filters = 1024,kernel_size = c(3,3), activation = 'relu',
                  kernel_initializer = 'he_normal',padding = 'same', data_format="channels_last", use_bias = FALSE) %>%
    layer_batch_normalization() 
  
  #Expansive Layers
  
  #E6
  classify<-c5 %>%
    layer_conv_2d_transpose(filters = 512, kernel_size=c(2,2), 
                            strides=c(2,2), padding = 'same', data_format="channels_last", use_bias = FALSE) %>%
    layer_batch_normalization() %>%
    {layer_concatenate(inputs= list (c4,.), axis=3)}%>%
    layer_conv_2d(filters =512,kernel_size = c(3,3),activation='relu', 
                  kernel_initializer='he_normal',padding='same', data_format="channels_last", use_bias = FALSE)%>%
    layer_batch_normalization() %>%
    layer_dropout(rate = FLAGS$dropout,seed = 1)%>%
    layer_conv_2d(filters =512,kernel_size = c(3,3),activation='relu', 
                  kernel_initializer='he_normal',padding='same', data_format="channels_last", use_bias = FALSE)%>%
    layer_batch_normalization() %>%
    
    #E7
    layer_conv_2d_transpose(filters =256,kernel_size= c(2,2), 
                            strides=c(2,2), padding = 'same', data_format="channels_last", use_bias = FALSE) %>%
    layer_batch_normalization() %>%
    {layer_concatenate(inputs= list (c3,.), axis=3)}%>%
    layer_conv_2d(filters =265,kernel_size = c(3,3),activation='relu', 
                  kernel_initializer='he_normal',padding='same', data_format="channels_last", use_bias = FALSE)%>%
    layer_batch_normalization() %>%
    layer_dropout(rate = FLAGS$dropout,seed = 1)%>%
    layer_conv_2d(filters =265,kernel_size = c(3,3),activation='relu', 
                  kernel_initializer='he_normal',padding='same', data_format="channels_last", use_bias = FALSE)%>%
    layer_batch_normalization() %>%
    
    #E8
    layer_conv_2d_transpose(filters =128, kernel_size=c(2,2), 
                            strides=c(2,2), padding = 'same', data_format="channels_last", use_bias = FALSE) %>%
    layer_batch_normalization() %>%
    {layer_concatenate(inputs= list (c2,.), axis=3)}%>%
    layer_conv_2d(filters =128,kernel_size = c(3,3),activation='relu', 
                  kernel_initializer='he_normal',padding='same', data_format="channels_last", use_bias = FALSE)%>%
    layer_batch_normalization() %>%
    layer_dropout(rate = FLAGS$dropout,seed = 1)%>%
    layer_conv_2d(filters =128,kernel_size = c(3,3),activation='relu', 
                  kernel_initializer='he_normal',padding='same', data_format="channels_last", use_bias = FALSE)%>%
    layer_batch_normalization() %>%
    
    #E9
    layer_conv_2d_transpose(filters =64, kernel_size=c(2,2), 
                            strides=c(2,2), padding = 'same', data_format="channels_last", use_bias = FALSE) %>%
    layer_batch_normalization() %>%
    {layer_concatenate(inputs= list (c1,.), axis=3)}%>%
    layer_conv_2d(filters =64,kernel_size = c(3,3),activation='relu', 
                  kernel_initializer='he_normal',padding='same', data_format="channels_last", use_bias = FALSE)%>%
    layer_batch_normalization() %>%
    layer_dropout(rate = FLAGS$dropout,seed = 1)%>%
    layer_conv_2d(filters =64,kernel_size = c(3,3),activation='relu', 
                  kernel_initializer='he_normal',padding='same', data_format="channels_last", use_bias = FALSE)%>%
    layer_batch_normalization() %>%
    
    # Output layer
    layer_conv_2d(filters =classes,kernel_size=c(1,1), activation = 'sigmoid',kernel_initializer='he_normal', data_format="channels_last")
  
  model <- keras_model(inputs = inputs, outputs = classify)
  
  return (model)
}
