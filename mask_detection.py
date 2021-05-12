import cv2, os
import matplotlib.pyplot as plt
import math
import random
#from PIL import Image
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from tensorflow.python.keras import models
import tensorflow as tf
from keras.models import load_model

def mask_states(file_path):
    file = open(file_path,'r')
    line = file.readline()
    line = file.readline().rstrip('\n').split(',')
    mask_state = []
    while line != ['']:
        target = 0
        if line[1] == 'without_mask':
            target = 1
        mask_state += [target]
        line = file.readline().rstrip('\n').split(',')
    file.close()
    random.shuffle(mask_state)
    return mask_state

def images_list(file_path):
    data_path = file_path
    data = []
    smallest = math.inf
    for image_name in os.listdir(data_path):
        image_path = os.path.join(data_path, image_name)
        img = cv2.imread(image_path)
        if img.shape[0] < smallest:
            smallest = img.shape[0]
        try:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            data.append(cv2.resize(gray,(100,100)))
        except Exception as e:
            print('Exception: ', e)
    return data

'''
# DATA PROCESSING  
'''

data = np.array(images_list('train'))/255.0
data = np.reshape(data,(data.shape[0],100,100,1))
target = np.array(mask_states('Training_set_face_mask.csv'))
# target = np_utils.to_categorical(target)

'''
# CONVOLUTIONAL ARCHITECTURE 
'''

model = Sequential()
# first layer
model.add(Conv2D(32,(3,3), input_shape=(100, 100,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#second layer
model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# flatten
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
# print(model.output_shape)
# print(model.summary())

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

history = model.fit(data,target,epochs=1,validation_split=0.2)


'''
# Testing 
''' 

testing_images = np.array(images_list('test'))/255.0
results = []
  
for image in testing_images: 
    temp = cv2.resize(image,(100,100))/255.0
    reshaped = np.reshape(temp,(1,100,100,1))
    probability = model.predict(reshaped)
    label = np.argmax(probability,axis=1)[0]
    L = 'without_mask'
    if label == 0:
        L = 'with_mask'
    results += [L]
    
print(results)





















