from __future__ import absolute_import, division, print_function

import os  
import PIL.Image as Image
import re
import operator
import random
import pathlib
import time

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

tf.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32

# Function recursivelly itterates through the path given to it, 'local_dir'.
  # For all of the files in the current directory, 
  # if the file has a ".jpg" extension, it is added to the list of absolute paths.
  #
  # local_dir - String - Absolute path to top level directory of pictures
  #
  # Returns - Array of strings, consisting of absolute paths, for all JPG pictures found
def get_image_paths(local_dir):
  paths = []
  for dirName, subDir, files in os.walk(local_dir):
    for f in files:
      found = re.search('\.jpg', f)
      if found:
        paths.append(os.path.join(dirName, f))
  return paths

# Function reads the JPG image, to transform it into a Tensor object. The individual pixels are
  # normalized to a range of [0,1].
  #
  # full_path - String - Absolute path of image to use
  # label - String - Default value "None." This is the label associated with the current image.
  #
  # Returns - Tensor object & label associated with it.
def preprocess_image(full_path, label=None): 
  img_raw = tf.read_file(full_path) # Get the raw data of the image.
  # All images appear to have "channels = 3" already. 
    # Transform inamge into a Tensor object.
    # The 'channels' attribute determines the color scheme to use:
    # 0 = Use the image's
    # 1 = Output a black & white image
    # 3 = Output a RGB image
  img = tf.image.decode_jpeg(img_raw, channels=1) # Got the argument: channels must be 0, 1, 3, or 4
  # Currently, all images have a height: 250 & width: 400, but had to force the values to work anyways.
  img = tf.image.resize_images(img, [250, 400])
  image  = img / 255.0  # normalize to [0,1] range
  return image, label

# Changes range from [0,1] to [-1,1]
  #
  # image - Tensor - Image that needs its pixels range changed
  # label - String - Label associated with current image
  #
  # Returns - The same image, and label, with an increased range.
def change_range(image,label):
  return 2*image-1, label

# Function is used to test the performance of the datasets.
def timeit(ds, batches=2*3+1):
  overall_start = time.time()
  # Fetch a single batch to prime the pipeline (fill the shuffle buffer),
  # before starting the timer
  it = iter(ds.take(batches+1))
  next(it)

  start = time.time()
  for i,(images,labels) in enumerate(it):
    if i%10 == 0:
      print('.',end='')
  print()
  end = time.time()

  duration = end-start
  print("{} batches: {} s".format(batches, duration))
  print("{:0.5f} Images/s".format(BATCH_SIZE*batches/duration))
  print("Total time: {}s".format(end-overall_start))

class_names = ['handgun', 'rifle', 'shotgun'] # List of available labels for the images
localTrainDir = os.environ['CS591LFDTRAINDIR'] # Location where the training image dataset is kept
localTestDir = os.environ['CS591LFDTESTDIR'] # Location where the test image dataset is kept

# Load the absolute paths into arrays to be used later
train_image_paths = get_image_paths(localTrainDir)
test_image_paths = get_image_paths(localTestDir)

# Assign an integer to the elements in the `class_names` vairable
class_to_index = dict((name, index) for index,name in enumerate(class_names))

# Randomize to help get rid of unwanted bias   
random.shuffle(train_image_paths)
train_image_count = len(train_image_paths)
# Use the directory the image is in as the label that is supposed to go with it
train_image_labels = [class_to_index[pathlib.Path(path).parent.name]
                    for path in train_image_paths]
# Randomize to help get rid of unwanted bias  
random.shuffle(test_image_paths)
test_image_count = len(test_image_paths)
# Use the directory the image is in as the label that is supposed to go with it
test_image_labels = [class_to_index[pathlib.Path(path).parent.name]
                    for path in test_image_paths]


##### Build Dataset
# Slicing the array of strings, results in a dataset of strings
train_path_ds = tf.data.Dataset.from_tensor_slices((train_image_paths, train_image_labels))
test_path_ds = tf.data.Dataset.from_tensor_slices((test_image_paths, test_image_labels))
# create a new dataset that loads and formats images on the fly by mapping preprocess_image over the dataset of paths.
train_image_label_ds = train_path_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
test_image_label_ds = test_path_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)



#### Starting to train
# https://www.tensorflow.org/tutorials/load_data/images#basic_methods_for_training

# Setting a shuffle buffer size as large as the dataset ensures that the data is
# completely shuffled.
ds = train_image_label_ds.shuffle(buffer_size=train_image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)


# `prefetch` lets the dataset fetch batches, in the background while the model is training.
ds = ds.prefetch(buffer_size=AUTOTUNE)

# mobile_net = tf.keras.applications.MobileNetV2(include_top=False)
# mobile_net.trainable=False
# Change the range from [0,1] to [-1,1]
keras_ds = ds.map(change_range)

# The dataset may take a few seconds to start, as it fills its shuffle buffer.
image_batch, label_batch = next(iter(keras_ds))

model = tf.keras.Sequential([
  # mobile_net,
  # tf.keras.layers.GlobalAveragePooling2D(),
  # tf.keras.layers.Dense(len(class_names))])
  keras.layers.Flatten(None, input_shape=(250, 400, 1)), # transforms the format of the images from a 2d-array (of 250 by 400 pixels), to a 1d-array of 250 * 400 = 100,000 pixels.
  keras.layers.Dense(128, activation=tf.nn.tanh), # layer has 128 nodes
  keras.layers.Dense(3, activation=tf.nn.softmax)]) # layer has 10 nodes. returns an array of 10 probability scores that sum to 1

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])

# model.fit(ds, epochs=5, steps_per_epoch=3)
model.fit(image_batch, label_batch, epochs=5, steps_per_epoch=3)



ds = train_image_label_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=train_image_count))
ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
print(ds)
timeit(ds)





