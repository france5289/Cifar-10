''' Classifier for cifar-10 '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
import gc
import load_data
import random

''' define parameters '''
BATCH_SIZE = 100
LEARNING_RATE = 0.01
STEPS = 2000
SEED = 66478

beginTime = time.time()

def weight_variable(shape, mean, stddev):
  return tf.Variable( tf.truncated_normal(shape, mean=mean, stddev=stddev, seed=SEED) )

def bias_variable(shape, value):
  return tf.Variable( tf.constant(value=value, shape=shape) )


''' Define the TensorFlow graph '''
images = tf.placeholder(tf.float32, shape=[None, 3072])
labels = tf.placeholder(tf.int64, shape=[None])
images_x = tf.reshape(images, [-1, 32, 32, 3]) # NHWC


# Convolution layer 1
conv1_Channel = 64
conv1_W = weight_variable([5, 5, 3, conv1_Channel], 0.0, 5e-2) # HWIO
conv1_b = bias_variable([conv1_Channel], 0.0)

conv1 = tf.nn.conv2d( images_x, conv1_W, strides=[1, 1, 1, 1], padding='SAME' )
bias1 = tf.nn.bias_add(conv1, conv1_b)

# Activation layer RELU
relu1 = tf.nn.relu(bias1)

# Activation layer MAXPOLLING
pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Convolution layer 2
conv2_Channel = 64
conv2_W = weight_variable([5, 5, conv1_Channel, conv2_Channel], 0.0, 5e-2)
conv2_b = bias_variable([conv2_Channel], 0.0)

conv2 = tf.nn.conv2d( pool1, conv2_W, strides=[1, 1, 1, 1], padding='SAME' )
bias2 = tf.nn.bias_add(conv2, conv2_b)
relu2 = tf.nn.relu(bias2)
pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# FLATTEN
f_in_size = pool2.get_shape().as_list()[1] # shape = (?, 8, 8, 64)
units = f_in_size * f_in_size * conv2_Channel # f_in_size = 8 = ( ( 32 / 2 ) / 2 )
flatten = tf.reshape(pool2, [-1, units])

# NN layer
nn1_W = weight_variable([units, 384], 0.0, 5e-2)
nn1_b = bias_variable([384], 0.1)

nn1_m = tf.matmul(flatten, nn1_W)
nn1 = tf.nn.bias_add(nn1_m, nn1_b)
relu3 = tf.nn.relu(nn1)

# NN layer
nn2_W = weight_variable([384, 192], 0.0, 5e-2)
nn2_b = bias_variable([192], 0.1)

nn2_m = tf.matmul(relu3, nn2_W)
nn2 = tf.nn.bias_add(nn2_m, nn2_b)
relu4 = tf.nn.relu(nn2)

# NN layer
nn3_W = weight_variable([192, 10], 0.0, 5e-2)
nn3_b = bias_variable([10], 0.1)

nn3_m = tf.matmul(relu4, nn3_W)
logits = tf.nn.bias_add(nn3_m, nn3_b)


''' Learning model, training weighting '''
# Loss function softmax - cross entropy
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

# Define the training operation
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)


''' For inference and calculating the accuracy of testing data '''
# Get prediction
pred = tf.argmax(logits, 1)

# Compare prediction with true label
correct_prediction = tf.equal(pred, labels)

# Calculate the accuracy of predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



''' Run the Tensorflow graph '''
with tf.Session() as sess:
  # Initialize variables
  sess.run(tf.global_variables_initializer())
  index = 0

  #Prepare train data
  train_data = load_data.load_data('Train')
  print('Train data images shape:', train_data['images'].shape)
  train_num = train_data['images'].shape[0]

  ''' Training '''
  for i in range(STEPS):
    # Generate train data batch
    index += BATCH_SIZE
    if (index + BATCH_SIZE > train_num):
      index = 0
    images_batch = train_data['images'][index : index+BATCH_SIZE]
    labels_batch = train_data['labels'][index : index+BATCH_SIZE]

    # Periodically print out the model's current accuracy
    if i % 100 == 0:
      train_accuracy = sess.run(accuracy, feed_dict={ images: images_batch, labels: labels_batch })
      print('Step {:6d}: training accuracy {:1.2f}'.format(i, train_accuracy))

    # Perform a single training step
    sess.run(train_step, feed_dict={images: images_batch, labels: labels_batch})
  del train_data
  gc.collect()


  #Prepare test data
  test_data = load_data.load_data('Test')
  print('Test data images shape:',test_data['images'].shape)
  test = 0.000

  ''' Testing '''
  for i in range(0, 100):
    index = i*100
    test_images = test_data['images'][index : index+100]
    test_labels = test_data['labels'][index : index+100]
    test_accuracy = sess.run(accuracy, feed_dict={ images: test_images, labels: test_labels})
    test += (test_accuracy*100)
    del test_images, test_labels
    gc.collect()
  test_accuracy = test/10000
  print('Test accuracy {:1.3f}'.format(test_accuracy))


  ''' Inference image '''
  number = random.randint(0, 9999) #input('Enter test image number: ')
  images_batch = test_data['images'][int(number):int(number)+1]
  labels_batch = test_data['labels'][int(number):int(number)+1]
  inference = sess.run(pred, feed_dict={ images: images_batch, labels: labels_batch })
  print('Prediction of image is', str(load_data.CLASSES[inference[0]]))
  load_data.show_image( test_data['images'][int(number)], test_data['labels'][int(number)] )
  

endTime = time.time()
print('Total time: {:5.2f}s'.format(endTime - beginTime))