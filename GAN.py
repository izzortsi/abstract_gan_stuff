# %%
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import load_img, image_dataset_from_directory
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Rescaling
from keras.layers import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
import sys
import os
import glob
import numpy as np

from PIL import Image
import os

data_path = os.path.join(os.getcwd(), 'data')
new_path = os.path.join(os.getcwd(), 'data_new')
output_path = os.path.join(os.getcwd(), 'output')
if not os.path.exists(new_path):
    os.makedirs(new_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)    
# %%

    
batch_size = 1
img_height = 16
img_width = 16
train_ds = image_dataset_from_directory(
  new_path,
  labels = None,
  label_mode=None,
#   validation_split=0.0,
#   subset="training",
  color_mode='rgb',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

load_img(
    os.path.join(new_path, '0000.jpg'),
    grayscale=False,
    color_mode='rgb',
    target_size=None,
    interpolation='nearest',
    keep_aspect_ratio=True,
)

# %%

normalization_layer = Rescaling(1./255)
normalized_ds = train_ds.map(lambda x: normalization_layer(x))
train_ds = iter(normalized_ds)

# %%
next(train_ds)
# %%
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(Conv2D(256, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[img_height, img_width, 3]))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1))

    return model


epochs = 1000
z_dim = [32, 32, 3]


# Noise for visualization
z_vis = tf.random.normal([4] + z_dim)

# %%

# Load data
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# x_train = x_train / 255.0
# x_iter = iter(tf.data.Dataset.from_tensor_slices(x_train).shuffle(4 * batch_size).batch(batch_size).repeat())
# next(x_iter)

# %%
# Generator
def make_generator():
  model = tf.keras.Sequential()
  # model.add(Conv2D(256, (5, 5), strides=(2, 2), padding='same',
  #                                    input_shape=z_dim))
  # model.add(BatchNormalization())
  # model.add(LeakyReLU())
  # model.add(Dropout(0.3))

  model.add(Conv2D(32, (5, 5), strides=(2, 2), padding='same'))
  model.add(BatchNormalization())
  model.add(LeakyReLU())
  model.add(Dropout(0.3))

  model.add(Conv2D(16, (5, 5), strides=(2, 2), padding='same'))
  model.add(BatchNormalization())
  model.add(LeakyReLU())
  model.add(Dropout(0.3))

  model.add(Flatten())
  model.add(Dense(img_height*img_width*3, activation='sigmoid'))
  model.add(Reshape((img_height, img_width, 3)))
  return model
# G = tf.keras.models.Sequential([
#   tf.keras.layers.Dense(28*28*3, input_shape = (z_dim, z_dim, 3), activation='relu'),
#   tf.keras.layers.Dense(28*28*3, activation='sigmoid'),
#   tf.keras.layers.Reshape((img_width, img_height, 3))])
G = make_generator()
# Discriminator
D = tf.keras.models.Sequential([
 tf.keras.layers.Flatten(input_shape=(img_width, img_height, 3)),
 tf.keras.layers.Dense(28*28*3 // 2, activation='relu'),
 tf.keras.layers.Dense(1)])

# Loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)
def G_loss(D, x_fake):
  return cross_entropy(tf.ones_like(D(x_fake)), D(x_fake))
def D_loss(D, x_real, x_fake):
  return cross_entropy(tf.ones_like(D(x_real)), D(x_real)) + cross_entropy(tf.zeros_like(D(x_fake)), D(x_fake))

# Optimizers
G_opt = tf.keras.optimizers.Adam(1e-4)
D_opt = tf.keras.optimizers.Adam(1e-4)

# %%

# Train
for epoch in range(epochs):
  z_mb = tf.random.normal([batch_size, *z_dim])
  x_real = next(train_ds)
  # Record operations
  with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:  
    x_fake = G(z_mb)
    G_loss_curr = G_loss(D, x_fake)
    D_loss_curr = D_loss(D, x_real, x_fake)
  # Gradients
  G_grad = G_tape.gradient(G_loss_curr, G.trainable_variables)
  D_grad = D_tape.gradient(D_loss_curr, D.trainable_variables)
  # Apply gradients
  G_opt.apply_gradients(zip(G_grad, G.trainable_variables))
  D_opt.apply_gradients(zip(D_grad, D.trainable_variables))
  
  if epoch % 100 == 0:
    # Print results
    print('epoch: {}; G_loss: {:.6f}; D_loss: {:.6f}'.format(epoch+1, G_loss_curr, D_loss_curr))
    # Plot generated images
    for i in range(1):
      plt.subplot(2, 5, i+1)
      plt.imshow(G(z_vis)[i,:,:]*255.0)
      plt.axis('off')
    plt.show()

# %%
