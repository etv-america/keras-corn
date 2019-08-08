#  minimal model setup and imports needed for our predictions to be done
import tensorflow as tf
import numpy as np
import os, time, cv2
tf.logging.set_verbosity(tf.logging.ERROR)  #  silences warnings

import keras
from keras import layers, models, optimizers
from keras.models import load_model

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 16} )
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

with tf.device("/gpu:0"):  #  can specify usse of CPUs instead of GPU if you're a masochist
    model = models.Sequential() 
    model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(256, 256, 3)))  #  image size taken by model
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5)) 
    model.add(layers.Dense(256, activation='relu'))  #  is 256 for the 2048 model, otherwise is usually 512, check model being used before running
    model.add(layers.Dense(1, activation='sigmoid')) 

model_name = '2048_large_drone_model'
model = load_model('./' + model_name + '.h5')                                  #  Info on current model:   trained over: full, unbalanced drone set 
model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['acc'])  #    (Update diligently)    learnrate: 1e-4,  batchsize: 32


predictions, is_pic = {}, ['jpg', 'jpeg', 'png']

while 1:  #  for now, loop forever
    files = os.listdir('.')  #  create list of the files present, and their respective names and file types
    filenames, extensions = [i.split('.')[0] for i in files], [i.split('.')[1] for i in files]

    for i in predictions.copy():  #  remove from the dictionary the images that have disappeared; copy is used to prevent failure when occurring live
        if i not in files:
            predictions.pop(i)
    
    for i in predictions:  #  temporary diagnostic dictionary printout, can replace with file write, export, etc.
        print(i,': ', predictions[i])
    print('--------')
    
    for i, val in enumerate(extensions):  
        if val.lower() in is_pic and filenames[i] not in predictions.keys():  #  check if a file is an image which has not already been predicted over
            image_data = cv2.resize(cv2.imread('./' + files[i]), (256, 256))
            formatted_imgd = np.expand_dims(np.array(image_data), axis=0)  #  formatting image for the predictor, broken up for easier debug
            predictions[files[i]] = int(model.predict(formatted_imgd)[0][0])
    time.sleep(3)  #  arbitrary wait to save memory/set pool, easily adjusted or superceded by a different limit (i.e. amount of images present)
