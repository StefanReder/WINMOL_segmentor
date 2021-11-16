
###Metrics

Precision <- custom_metric("Precision", function (y_true,y_pred) {
  y_pred <- k_round(y_pred)
  k_sum(y_pred*y_true)/(k_sum(y_pred)+k_epsilon())
}) 

Recall <- custom_metric("Recall", function (y_true,y_pred) {
  y_pred <- k_round(y_pred)
  k_sum(y_pred*y_true)/(k_sum(y_true)+k_epsilon())
}) 

F1Score <- custom_metric("F1Score", function (y_true,y_pred) {
  y_pred <- k_round(y_pred)
  precision <- k_sum(y_pred*y_true)/(k_sum(y_pred)+k_epsilon())
  recall    <- k_sum(y_pred*y_true)/(k_sum(y_true)+k_epsilon())
  (2*precision*recall)/(precision+recall+k_epsilon())}) 

###Loss Function

K <- backend()
F1Score_loss <- function(y_true, y_pred) {
  result <- loss_binary_crossentropy(y_true, y_pred) +
    (1 - F1Score(y_true, y_pred))
  return(result)
}

###Optimizer

optimizer <- optimizer_adam(
  learning_rate = 1e-3,
  beta_1 = 0.9,
  beta_2 = 0.999,
  epsilon = NULL,
  decay = 0,
  amsgrad = FALSE,
  clipnorm = NULL,
  clipvalue = NULL)
