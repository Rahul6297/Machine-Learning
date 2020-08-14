# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 17:40:39 2018

@author: Mridul Gupta
         18EC65R10
         MTech 1st Year VIPES
"""

from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator


#Convolution Neural network
model = models.Sequential()
#Convolutional Layer 1 No. of Filters=32 Filter=3x3 Input size=150x150x3 
model.add(layers.Conv2D(32, (3, 3), activation='relu',
input_shape=(150, 150, 3)))
#MaxPool Layer 1
model.add(layers.MaxPooling2D((2, 2)))
#Convolutional Layer 2 No. of Filters=64 Filter=3x3x32 Input size =74x74x32 
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#MaxPool Layer 2
model.add(layers.MaxPooling2D((2, 2)))
#Convolutional Layer 3 No. of Filters=128, Filter=3x3x64 Input size =36x36x64 
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
#MaxPool Layer 3
model.add(layers.MaxPooling2D((2, 2)))
#Convolutional Layer 4 No. of Filters=128, Filter=3x3x64 Input size =17x17x128
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
#MaxPool Layer 4
model.add(layers.MaxPooling2D((2, 2)))
#Flatten Layer
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
#Hidden Layer
model.add(layers.Dense(512, activation='relu'))
#Classification layer
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

#Locations of trianing, validation and test data folder 
train_dir='D:\\IIT Kharagpur Journey\\PROJECT DIP\\Poor Index_Sorted Data\\train'
validation_dir='D:\\IIT Kharagpur Journey\\PROJECT DIP\\Poor Index_Sorted Data\\validation'
test_dir='D:\\IIT Kharagpur Journey\\PROJECT DIP\\Poor Index_Sorted Data\\test'

# Data Augmentation configuration via ImageData Generator
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255)

#Training Data
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=32,
                                                    class_mode='binary')

#Note that the validation data shouldn’t be augmented!
validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size=(150, 150),#Resizes all images to 150 × 150
                                                        batch_size=32,
                                                        class_mode='binary')

#Training the data for given no of Epochs(iterations on training dataset)
history = model.fit_generator(train_generator,
                              steps_per_epoch=20,
                              epochs=100,
                              validation_data=validation_generator,
                              validation_steps=50)

#Test Data
test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(150, 150),
                                                  batch_size=20,
                                                  class_mode='binary')
# Test loss and Accuracy
test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)
print('test loss:', test_loss)

#Saving the model, to use it at later stage
model.save('Poor Index_2.h5')


import matplotlib.pyplot as plt
#Training Accuracy
acc = history.history['acc']
#Validation Accuracy
val_acc = history.history['val_acc']
#Training data loss
loss = history.history['loss']
#Validation data loss
val_loss = history.history['val_loss']
#Number rof Epochs
epochs = range(1, len(acc) + 1)

#Plotting the differeent curves
#Plotting Traning Accuracy vs Validation Accuracy Curve
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

#Plotting Training Loss vs Validation Loss curve
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()