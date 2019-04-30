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
import os

model_path = './model'

accuracy = 0
with tf.Session() as sess:
  saver = tf.train.import_meta_graph(model_path+'/cifar10-model.ckpt.meta')
  saver.restore(sess, tf.train.latest_checkpoint(model_path))
    
  graph = tf.get_default_graph()

  images = graph.get_tensor_by_name("x:0")
  labels = graph.get_tensor_by_name("y:0")
  pred = graph.get_tensor_by_name("pred:0")


  test_data = load_data.load_data('Test')
  num_test = len(test_data['labels'])

  ''' Inference image '''
  number = random.randint(0, 9999)
  images_batch = test_data['images'][int(number):int(number)+1]
  labels_batch = test_data['labels'][int(number):int(number)+1]
  inference = sess.run(pred, feed_dict={ images: images_batch, labels: labels_batch })
  print('Prediction of image is', str(load_data.CLASSES[inference[0]]))
  load_data.show_image( test_data['images'][int(number)], test_data['labels'][int(number)] )

  '''
  for i in range(num_test):
    images_batch = test_data['images'][int(i):int(i)+1]
    labels_batch = test_data['labels'][int(i):int(i)+1]
    inference = sess.run(pred, feed_dict={ images: images_batch, labels: labels_batch })
    if inference[0] == test_data['labels'][int(i)]:
        accuracy += 1
  print('Test accuracy {:1.3f}'.format(accuracy/num_test))
  '''
