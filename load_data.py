''' Load CIFAR-10 dataset '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time
import pickle
import sys
import gc
from random import random
from PIL import Image

DATA_DIR = './cifar-10-batches-py'
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
DEPTH = 3 
HEIGHT = 32
WIDTH = 32

SHOW_INDEX = 1

def build_image( h, w, dd, label ):
  dd = dd * 255
  data = dd.astype(np.uint8).reshape( ( h, w, 3 ) ) # dtype is uint8, channel is 3
  img = Image.fromarray(data, 'RGB').resize((160, 160))
  #file_name = str(index) + ".png"
  #img.save( 'image/'+file_name )
  img.title = CLASSES[label]
  print('True label is', str(img.title))
  img.show()


def show_image(data, label):
  image = []
  for i in range(0, 1024):
    rgb = [ data[i], data[1024+i], data[2048+i] ]
    image.append(rgb)
  build_image( 32, 32, np.array(image), int(label) )


def shuffle_data( data, labels ):
  shuffleX = np.arange(len(data))
  np.random.shuffle(shuffleX)
  data = data[shuffleX]
  labels = labels[shuffleX]
  del shuffleX
  gc.collect()
  return data, labels


def normalize_data( data ):
  return data.astype(np.float) / 255.0


def unpickle(file, shuffle):
  with open(file, 'rb') as fo:
      dict = pickle.load(fo, encoding='bytes')
  x = np.array( dict[b'data'] )
  y = np.array( dict[b'labels'] )

  # Normalize Data
  x = normalize_data( x )
  if (shuffle == 'True'):
    return shuffle_data( x, y )
  else:
    return x, y


def load_data(data):
  t1 = time.time()

  X = []
  Y = []
  # prepare train data
  if (data == 'Train'):
    for i in range(1, 6): # 1...5
      filename = 'data_batch_' + str(i)
      filename = os.path.join(DATA_DIR, filename)
      load_X, load_Y = unpickle( filename, 'True' )
      X = np.append(X, load_X)
      Y = np.append(Y, load_Y)

      del load_X, load_Y
      gc.collect()
      print('load', filename, 'finish')

    X = X.reshape((-1, DEPTH * HEIGHT * WIDTH))
  # prepare test data
  elif (data == 'Test'):
    X, Y = unpickle( os.path.join(DATA_DIR, 'test_batch'), 'False' )
  else:
    print('Data load error, please command Train or Test data')

  #show_image(X[SHOW_INDEX,], Y[SHOW_INDEX])

  data_dict = {
    'images': X,
    'labels': Y,
    'classes': CLASSES
  }

  del X, Y
  gc.collect()
  t2 = time.time()
  print('Prepare data use {:5.2f}s'.format(t2 - t1))
  return data_dict
  

if __name__ == '__main__':
  train_data = load_data('Train')
  print(train_data['images'].shape)
  print(train_data['labels'].shape)

  test_data = load_data('Test')
  print(test_data['images'].shape)
  print(test_data['labels'].shape)

