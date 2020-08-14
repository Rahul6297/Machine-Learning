# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 01:23:36 2018
@author: Mridul Gupta
         18EC65R10
         MTech 1st Year VIPES
"""
#Instantiating the VGG16 convolutional base
from keras.applications import VGG16 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

#Summary of VGG16 model
conv_base.summary()

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
#Base directory of all the Data
base_dir = 'D:\\IIT Kharagpur Journey\\PROJECT DIP\\Latest Sorted Dataset\\Poor+Rich Data'
train_dir = os.path.join(base_dir, 'Train')
validation_dir = os.path.join(base_dir, 'Validation')
test_dir = os.path.join(base_dir, 'Test')
datagen = ImageDataGenerator(rescale=1./255)

batchsize = 20

#A function to Extract Features using pretrained convolutional base(VGG16)
def extract_features(directory, sample_count):
    #Initialising with Zeros
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(directory,target_size=(150, 150),
                                            batch_size=batchsize,
                                            class_mode='binary')
    #print (features)
    #print (labels)
    i = 0
    # You’ll extract features from these images 
    # by calling the predict method of the conv_base model
    print("Before generator loop")
    for inputs_batch, labels_batch in generator:
        #print("loop",i)
        features_batch = conv_base.predict(inputs_batch)
        features[i * batchsize : (i + 1) * batchsize] = features_batch
        labels[i * batchsize : (i + 1) * batchsize] = labels_batch
       # print(features_batch)
        #print(labels_batch)
        i += 1
        if i * batchsize >= sample_count:
            # Because generators yield data indefinitely in a loop,
            #we must break after every image has been passed once.
            break 
    return features, labels

train_features, train_labels = extract_features(train_dir, 640) 
#print (train_features)
#print (train_labels)
validation_features, validation_labels = extract_features(validation_dir, 100)
test_features, test_labels = extract_features(test_dir, 160)

#The extracted features are currently of shape (samples, 4, 4, 512). 
#You’ll feed them to a densely connected classifier, so first you must flatten them to (samples, 8192)
train_features = np.reshape(train_features, (640, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (100, 4 * 4 * 512))
test_features = np.reshape(test_features, (160, 4 * 4 * 512))

#Defining and training the densely connected classifier
from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, 
                    train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))

model.summary()

#Plotting the results
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)


test_loss, test_acc = model.evaluate(test_features, test_labels)
print('test_acc:', test_acc)
print('test_loss:', test_loss)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()