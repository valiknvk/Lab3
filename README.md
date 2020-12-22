#Скрипт для обучения

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.keras import backend
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, load_model
from keras.models import model_from_json
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, Masking, TimeDistributed
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.preprocessing import image
import keras
from keras.engine import Layer
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, merge, concatenate
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard 
from tensorflow.keras.models import Sequential, Model
from keras.layers.core import RepeatVector, Permute
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import os#, cv2 
import random
from datetime import datetime


#tf.compat.v1.enable_eager_execution()
#import tensorflow_io as tfio
from absl import app
import random
import threading
import matplotlib.pyplot as plt

global graph, model
graph = tf.compat.v1.get_default_graph()
tf.compat.v1.disable_eager_execution()

path = 'D:\\pictures\\test\\'
filenames = list()
n = 0
for file in os.listdir(path):
  filenames.append(path+file)
  n+=1
  if (n == 3000):
    break
print(len(filenames))  
lostTrBatch = list()
lostEp = list()
lostTestBatch = list()



class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):

  def on_train_batch_end(self, batch, logs=None):
    print('For batch {}, loss is {:7.2f}.'.format(batch, logs['loss']))
    lostTrBatch.append(logs['loss'])
  def on_test_batch_end(self, batch, logs=None):
    print('For batch {}, loss is {:7.2f}.'.format(batch, logs['loss']))
    lostTestBatch.append(logs['loss'])
  def on_epoch_end(self, epoch, logs=None):
    print('Средние потери за эпоху {} равны {:7.2f}'.format(epoch, logs['loss']))
    lostEp.append(logs['loss'])
X = []
for filename in filenames:
    X.append(img_to_array(load_img(filename)))
X = np.array(X, dtype=float)# Set up training and test data
split = int(0.95*len(X))
Xtrain = X[:split]
Xtrain = 1.0/255*Xtrain#Design the neural network
model = Sequential()
model.add(InputLayer(input_shape=(256, 256, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))# Finish model
model.compile(optimizer='rmsprop', loss='mse')# Image transformer
datagen = ImageDataGenerator(
        #shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)# Generate training data
batch_size = 5#20
def image_a_b_gen(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)# Train model
#logdir = 'D:\\pictures\\log\\fit\\' + datetime.now().strftime('%Y%m%d-%H%M%S')

#TensorBoard(log_dir=logdir)
#tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

model.fit(image_a_b_gen(batch_size), steps_per_epoch=10, epochs=30, callbacks=[LossAndErrorPrintingCallback()])#,verbose=0, callbacks=[LossAndErrorPrintingCallback()])
# Test images

Xtest = rgb2lab(1.0/255*X[split:])[:,:,:,0]
Xtest = Xtest.reshape(Xtest.shape+(1,))
Ytest = rgb2lab(1.0/255*X[split:])[:,:,:,1:]
Ytest = Ytest / 128
print (model.evaluate(Xtest, Ytest, batch_size=batch_size))# Load black and white images
print(model.summary())
# serialize model to JSON
MyModel = model.to_json()
with open("D:\\pictures\\model.json", "w") as json_file:
    json_file.write(MyModel)
# serialize weights to HDF5
model.save_weights("D:\\pictures\\model.h5")
color_me = []
n = 0
for filename in filenames:
        
        #if (n > 0):
        #  color_me.append(img_to_array(load_img(filename)))
        #if (n>50):
        #  continue
        #if (n > 1500):
        color_me.append(img_to_array(load_img(filename)))
        n+=1
        if (n==50):
          break
color_me = np.array(color_me, dtype=float)
color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
color_me = color_me.reshape(color_me.shape+(1,))# Test model
output = model.predict(color_me)
output = output * 128# Output colorizations
check = output[len(output)-1]
#plt.imshow(lab2rgb(check))
for i in range(len(output)):
        cur = np.zeros((256, 256, 3))
        cur[:,:,0] = color_me[i][:,:,0]
        cur[:,:,1:] = output[i]
        imsave("D:\\pictures\\50learned\\"+"img_"+str(i)+".png", lab2rgb(cur))
        #plt.imshow(lab2rgb(cur))
        #imsave("img_"+str(i)+".png", lab2rgb(cur))
#f = open('D:\\pictures\\loss.txt', 'w')
#for i in lostEp:
#    print (str(i))
#    f.write( str(i))
#    f.write('\n')
#f.close
plt.plot(np.array(lostEp))
plt.savefig('D:\\pictures\\Loss_Val.png')
