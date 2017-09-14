from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Activation, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import numpy as np

width, height = 150, 150


training_path = "/Users/sominwadhwa/Work/data/DogsVsCats/train"
val_path = "/Users/sominwadhwa/Work/data/DogsVsCats/val"
n_train = 2000
n_val = 400
epochs = 50
batch_size = 32

if K.image_data_format()=='channels_first':
    input_shape = (3, width, height)
else:
    input_shape = (width, height, 3)

model = Sequential()
model.add(Conv2D(32,(3,3), input_shape= input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])

train_data = ImageDataGenerator(
        rescale= 1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_data = ImageDataGenerator(rescale=1./255)

train = train_data.flow_from_directory(
        training_path,
        class_mode='binary',
        batch_size=batch_size,
        target_size=(width,height))

validation = test_data.flow_from_directory(
        val_path,
        class_mode='binary',
        batch_size=batch_size,
        target_size=(width,height))

filepath = "weight-improv-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode = 'max')
callbacks_list=[checkpoint]


model.fit_generator(
        train,
        steps_per_epoch = n_train // batch_size,
        epochs = epochs,
        validation_data = validation,
        validation_steps = n_val // batch_size,
        callbacks= callbacks_list
        )
model.save_weights('very_little_weights.h5')





