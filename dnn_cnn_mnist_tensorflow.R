rm(list = ls()); gc(reset = T)

library(tensorflow)

input_dataset <- tf$examples$tutorials$mnist$input_data
mnist <- input_dataset$read_data_sets("MNIST-data", one_hot = TRUE)

sess <- tf$InteractiveSession()

x <- tf$placeholder(tf$float32, shape(NULL, 784L))
y <- tf$placeholder(tf$float32, shape(NULL, 10L))

weight <- function(shape) {
  
  tf$Variable(tf$truncated_normal(shape, stddev = 0.1))
  
}

bias <- function(shape) {
  
  tf$Variable(tf$constant(0.1, shape = shape))
  
}

conv2d <- function(x, W) {
  
  tf$nn$conv2d(x, W, strides=c(1L, 1L, 1L, 1L), padding = 'SAME')
  
}

max_pool <- function(x) {
  
  tf$nn$max_pool(x, ksize=c(1L, 2L, 2L, 1L), strides=c(1L, 2L, 2L, 1L), padding = 'SAME')
  
}

W_conv1 <- weight(shape(5L, 5L, 1L, 32L))
b_conv1 <- bias(shape(32L))

x_image <- tf$reshape(x, shape(-1L, 28L, 28L, 1L))

h_conv1 <- tf$nn$relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 <- max_pool(h_conv1)

W_conv2 <- weight(shape = shape(5L, 5L, 32L, 64L))
b_conv2 <- bias(shape = shape(64L))

h_conv2 <- tf$nn$relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 <- max_pool(h_conv2)

W_fc1 <- weight(shape(7L * 7L * 64L, 1024L))
b_fc1 <- bias(shape(1024L))

h_pool2_flat <- tf$reshape(h_pool2, shape(-1L, 7L * 7L * 64L))
h_fc1 <- tf$nn$relu(tf$matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob <- tf$placeholder(tf$float32)
h_fc1_drop <- tf$nn$dropout(h_fc1, keep_prob)

W_fc2 <- weight(shape(1024L, 10L))
b_fc2 <- bias(shape(10L))

y_conv <- tf$nn$softmax(tf$matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy <- tf$reduce_mean(-tf$reduce_sum(y * tf$log(y_conv), reduction_indices=1L))

train_step <- tf$train$AdamOptimizer(1e-4)$minimize(cross_entropy)

correct_prediction <- tf$equal(tf$argmax(y_conv, 1L), tf$argmax(y, 1L))

accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))

sess$run(tf$global_variables_initializer())

for (i in 1:100) {
  
  batch <- mnist$train$next_batch(100L)
  
  if (i %% 100 == 0) {
    
    train_accuracy <- accuracy$eval(feed_dict = dict(x = batch[[1]], 
                                                     y = batch[[2]], keep_prob = 1.0))
    cat(sprintf("step %d, training accuracy %g\n", i, train_accuracy))
    
  }
  train_step$run(feed_dict = dict(x = batch[[1]], y = batch[[2]], keep_prob = 0.5))
}

test_accuracy <- accuracy$eval(feed_dict = dict(x = mnist$test$images, 
                                                y = mnist$test$labels, keep_prob = 1))
cat(sprintf("test accuracy %g", test_accuracy))