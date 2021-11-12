###Callbacks

callbacks_train1 <- list(
  callback_model_checkpoint(filepath=paste(
    checkdir,model_name,"_",dataset_name,"_",format(Sys.time(),"%Y-%m-%d_%H%M%S"),"_weights.train1.{epoch:02}-{val_loss:.02}.hdf5",sep=""),
    monitor = "val_loss", verbose=1, save_best_only=TRUE),
  callback_reduce_lr_on_plateau(monitor = "val_loss",
                                factor = 0.1,
                                patience = 2,
                                verbose = 1,
                                mode = "min",
                                min_delta = 1e-04,
                                cooldown = 0,
                                min_lr = 0),
  callback_early_stopping(monitor = "val_loss",
                          mode="min",
                          patience = 3,
                          verbose = 1),    
  callback_tensorboard(log_dir=paste(logdir,model_name,"_",dataset_name,"_train1_",format(Sys.time(),"%Y-%m-%d_%H%M%S"),'/',sep=""),histogram_freq=1, 
                       write_graph=TRUE,
                       write_images=TRUE,
                       update_freq='epoch',
                       profile_batch=2,
                       embeddings_freq=1)
)

callbacks_train2 <- list(
  callback_model_checkpoint(filepath=paste(
    checkdir,model_name,"_",dataset_name,"_",format(Sys.time(),"%Y-%m-%d_%H%M%S"),"_weights.train2.{epoch:02}-{val_loss:.02}.hdf5",sep=""),
    monitor = "val_loss", verbose=1, save_best_only=TRUE),
  callback_reduce_lr_on_plateau(monitor = "val_loss",
                                factor = 0.1,
                                patience = 2,
                                verbose = 1,
                                mode = "min",
                                min_delta = 1e-04,
                                cooldown = 0,
                                min_lr = 0),
  callback_early_stopping(monitor = "val_loss",
                          mode="min",
                          patience = 5,
                          verbose = 1),   
  callback_tensorboard(log_dir=paste(logdir,model_name,"_",dataset_name,"_train2_",format(Sys.time(),"%Y-%m-%d_%H%M%S"),'/',sep=""),histogram_freq=1, 
                       write_graph=TRUE,
                       write_images=TRUE,
                       update_freq='epoch',
                       profile_batch=2,
                       embeddings_freq=1))