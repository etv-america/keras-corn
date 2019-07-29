#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# Greetings from the Kaggle bot! This is an automatically-generated kernel with starter code demonstrating how to read in the data and begin exploring. Click the blue "Edit Notebook" or "Fork Notebook" button at the top of this kernel to begin editing.

# ## Exploratory Analysis
# To begin this exploratory analysis, first use `matplotlib` to import libraries and define functions for plotting the data. Depending on the data, not all plots will be made. (Hey, I'm just a kerneling bot, not a Kaggle Competitions Grandmaster!)

# In[27]:


import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

# To see our directory

import os
import random
import gc # Garbage collector for cleaning deleted data from memory


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# There is 0 csv file in the current version of the dataset:
# 

# In[28]:


train_healthy_dir = './input/dataset/train_corn/healthy/'
train_spot_dir = './input/dataset/train_corn/spot/'
train_rust_dir = './input/dataset/train_corn/rust/'
train_blight_dir = './input/dataset/train_corn/blight/'

test_healthy_dir = './input/dataset/test_corn/healthy/'
test_spot_dir = './input/dataset/test_corn/spot/'
test_rust_dir = './input/dataset/test_corn/rust/'
test_blight_dir = './input/dataset/test_corn/blight/'

train_healthy = [train_healthy_dir+'{}'.format(i) for i in os.listdir(train_healthy_dir)]
train_spot = [train_spot_dir+'{}'.format(i) for i in os.listdir(train_spot_dir)]
train_rust = [train_rust_dir+'{}'.format(i) for i in os.listdir(train_rust_dir)]
train_blight = [train_blight_dir+'{}'.format(i) for i in os.listdir(train_blight_dir)]

test_healthy = [test_healthy_dir+'{}'.format(i) for i in os.listdir(test_healthy_dir)]
test_spot = [test_spot_dir+'{}'.format(i) for i in os.listdir(test_spot_dir)]
test_rust = [test_rust_dir+'{}'.format(i) for i in os.listdir(test_rust_dir)]
test_blight = [test_blight_dir+'{}'.format(i) for i in os.listdir(test_blight_dir)]



# The next hidden code cells define functions for plotting data. Click on the "Code" button in the published kernel to reveal the hidden code.

# In[29]:


train_imgs = train_blight[:len(train_blight)-1] + train_healthy[:len(train_blight)-1]
random.shuffle(train_imgs)

gc.collect()


# In[30]:


import matplotlib.image as mpimg
for ima in train_imgs[0:3]:
    img = mpimg.imread(ima)
    imgplot = plt.imshow(img)
    plt.show()

    


# In[31]:


nrows = 256
ncolumns = 256
channels = 3


# In[32]:


print(train_healthy[:5])


# In[33]:


print(train_imgs[:5])


# In[34]:


def read_and_process_image(list_of_images):
    x = []
    y = []
    
    for image in list_of_images:
        x.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows, ncolumns), interpolation=cv2.INTER_CUBIC))
        
        if image in train_healthy:
            y.append(1)
        else:
            y.append(0)
            
    return x, y

x, y = read_and_process_image(train_imgs)


# In[35]:




plt.figure(figsize=(20, 10))
columns = 5
for i in range(columns):
    print(i,"is a", y[i])
    plt.subplot(5 / columns + 1, columns, i + 1)
    plt.imshow(x[i])


# In[36]:


import seaborn as sns
gc.collect()

x = np.array(x)
y = np.array(y)

sns.countplot(y)
plt.title('Labels for Healthy and Blight')


# In[37]:


print("Shape of train images is:", x.shape)
print("Shape of labels is: ", y.shape)
print(x)


# In[38]:


test_imgs = test_blight[:len(test_blight)-1] + test_healthy[:len(test_blight)-1]
random.shuffle(test_imgs)

gc.collect()


# In[39]:


ntrain = len(train_imgs)
nval = len(test_imgs)

batch_size = 32

import tensorflow as tf

# In[40]:

import keras
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img


# In[41]:
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess= tf.Session(config=config)
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)

with tf.device("/gpu:0"):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(256, 256, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))  #Dropout for regularization
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  #Sigmoid function at the end because we have just two classes


    # In[42]:


    model.summary()


    # In[43]:


    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])


    # In[44]:


    #Lets create the augmentation configuration
    #This helps prevent overfitting, since we are using a small dataset
    train_datagen = ImageDataGenerator(rescale=1./255,   #Scale the image between 0 and 1
                                        rotation_range=40,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True,)

    val_datagen = ImageDataGenerator(rescale=1./255)  #We do not augment validation data. we only perform rescale


# In[45]:


def read_and_process_image_test(list_of_images):
    X = []
    Y = []
    
    for image in list_of_images:
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows, ncolumns), interpolation=cv2.INTER_CUBIC))
        
        if image in test_healthy:
            Y.append(1)
        else:
            Y.append(0)
            
    return X, Y

x_test, y_test = read_and_process_image_test(test_imgs)
x_test = np.array(x_test)
y_test = np.array(y_test)


# In[46]:


#x, y = read_and_process_image_test(train_imgs)


# In[47]:



#Create the image generators
train_generator = train_datagen.flow(x, y, batch_size=batch_size)
val_generator = val_datagen.flow(x_test, y_test, batch_size=batch_size)


# In[48]:


print(x.shape)


# In[49]:


# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


# In[50]:


# from keras import backend as K
# K.tensorflow_backend._get_available_gpus()


# In[51]:


# In[52]:

# In[53]:


#The training part
#We train for 64 epochs with about 100 steps per epoch

history = model.fit_generator(train_generator,
                                  steps_per_epoch=ntrain // batch_size,
                                  epochs=64,
                                  validation_data=val_generator,
                                  validation_steps=nval // batch_size)

