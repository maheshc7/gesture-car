"""
Created on Sat Nov 25 00:20:41 2021

@author: ananya
"""
#required packages

import pandas as pd
import numpy as np 
import itertools
import keras
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
from keras.models import Sequential 
from tensorflow.keras import optimizers
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense  
from keras import applications  
from keras.utils.np_utils import to_categorical  
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import math  
import datetime
import time
import os
import tensorflow as tf

#%%

img_width, img_height = 224, 224  
num_classes = 5 
   
epochs = 50 
batch_size = 64

#train-test split

IMAGE_HEIGHT=img_height
IMAGE_WIDTH= img_width

#%%
#preparing dataset
def get_pathframe(path):
  '''
  Get all the images paths and its corresponding labels
  Store them in pandas dataframe
  '''
  categories = []
  paths=[]
  dirnames = os.listdir(path)
  for dir in dirnames:
    dir +="/"
    filenames = os.listdir(path+dir)
    for filename in filenames:
      paths.append(path+dir+filename)
      category = dir.split('/')[0]
      if category == 'forward':
        categories.append(0)
      elif category == 'stop':
        categories.append(4)
      elif category == 'reverse':
        categories.append(2)
      elif category == 'left':
        categories.append(1)
      elif category == 'right':
        categories.append(3)

  df= pd.DataFrame({
      #'filename': filenames,
      'category': categories,
      'paths':paths
  })
  return df


def load_and_preprocess_image(path):
  '''
  Load each image and resize it to desired shape
  '''
  image = tf.io.read_file(path)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [IMAGE_WIDTH, IMAGE_HEIGHT])
  image /= 255.0  # normalize to [0,1] range
  return image

def convert_to_tensor(df):
  '''
  Convert each data and labels to tensor
  '''
  path_ds = tf.data.Dataset.from_tensor_slices(df['paths'])
  image_ds = path_ds.map(load_and_preprocess_image)
  # onehot_label=tf.one_hot(tf.cast(df['category'], tf.int64),2) if using softmax
  onehot_label=tf.cast(df['category'], tf.int64)
  label_ds = tf.data.Dataset.from_tensor_slices(onehot_label)
  print(label_ds)

  return image_ds,label_ds


X,Y=convert_to_tensor(df)

dataset=tf.data.Dataset.zip((X,Y)).shuffle(buffer_size=2000)
dataset_train=dataset.take(2300)
dataset_test=dataset.skip(513)

#print(dataset)
#print(dataset_train)
#print(dataset_test)

dataset_train=dataset_train.batch(batch_size, drop_remainder=True)
dataset_test=dataset_test.batch(batch_size, drop_remainder=True)
#dataset_train
#dataset_test
#%%
#pretrained VGG-16 model
from tensorflow.keras.applications import VGG16
base_model = VGG16(input_shape=(img_width,img_height,3), include_top=False, weights="imagenet")


#%%
#transfer learning on VGG16
from tensorflow.keras import layers
for layer in base_model.layers[:15]:
    layer.trainable = False

for layer in base_model.layers[15:]:
    layer.trainable = True

    
last_layer = base_model.get_layer('block5_pool')
last_output = last_layer.output
x = layers.Flatten()(last_output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(num_classes, activation='softmax')(x)

vggmodel = tf.keras.models.Model(base_model.input, x)

vggmodel.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

vggmodel.summary()

#%%
#training model
vgg16_final=vggmodel.fit(dataset_train,epochs=50,validation_data=dataset_test, batch_size=batch_size)

# %%
#Arduino Code
import serial
import time

# Define the serial port and baud rate.
# Ensure the 'COM#' corresponds to what was seen in the Windows Device Manager
ser = serial.Serial('COM5', 9600)

def run_car(command):
    if command =="Forward":
        ser.write(b'w')
    elif command =="Reverse":
        ser.write(b's')
    elif command =="Left":
        ser.write(b'd')
    elif command =="Right":
        ser.write(b'a')
    elif command =="Stop":
        ser.write(b'e')
    else:
        print("Invalid input")
#%%
import cv2
path="C:/Anan/MS/projects/AI530/Sign Language Project/"
vggmodel.load_weights('C:/Anan/MS/projects/AI530/Sign Language Project/weights/vgg_model_v2.weights.best.hdf5')
classes = ["Forward","Left","Reverse","Right","Stop"]
pred_class = ""
vid = cv2.VideoCapture(0)
frameno, imgno=1,1
while(True):
    
    ret, frame = vid.read()
    if(ret==True):
        cv2.imshow('frame', frame)    
        frameno+=1
        if(frameno%24==0):
            imgno+=1
            cv2.imwrite(path+"frames/"+str(imgno)+".jpg",frame)
            img = tf.keras.preprocessing.image.load_img(path+"frames/"+str(imgno)+".jpg",target_size=(224,224))
            img_nparray = tf.keras.preprocessing.image.img_to_array(img)
            type(img_nparray) 
            input_Batch = np.array([img_nparray])  
            vgg_pred = vggmodel.predict(input_Batch)
            vgg_pred_classes= np.argmax(vgg_pred, axis=1)
            if pred_class != classes[vgg_pred_classes[0]]:
                pred_class = classes[vgg_pred_classes[0]]
                print("New Command",pred_class)
                run_car(pred_class)
                #time.sleep(1)
                #print(vgg_pred,vgg_pred_classes,pred_class)
            
            frameno=0
                       
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()