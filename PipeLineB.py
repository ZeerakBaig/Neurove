# image classification using CNN (Deep learning)
import os
import glob
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from keras.models import Sequential, Model, load_model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import Callback, EarlyStopping
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from keras import backend as K
from sklearn.metrics import classification_report
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import seaborn as sns

# create the folders for train validation and testing
# create 3 folder
from tensorflow.python.keras.constraints import maxnorm

# we have imported open CV for image classification and will be using keras with tensorflow as backend


data_directory = Path("C:/Users/User/Desktop/My Courses/Honours/Biometrics/Project/Final_Data_Set/")
# we split the dataset into 3 sets
train_directory = data_directory / 'train'
validation_directory = data_directory / 'val'
test_directory = data_directory / 'test'


# function that returns creates a dataframe with 2 columns ->
# one representing the path to the image and the second one is a label
def load_train():
    normal_faces_directory = train_directory / 'Healthy'
    paralysis_faces_directory = train_directory / 'UnHealthy'
    # getting the list of all the images
    normal_faces = normal_faces_directory.glob('*.jpg')
    paralysis_faces = paralysis_faces_directory.glob('*.*')

    # adding the dta to the list
    train_data = []
    train_label = []
    for img in normal_faces:
        train_data.append(img)
        train_label.append('HEALTHY')
    for img in paralysis_faces:
        train_data.append(img)
        train_label.append('UNHEALTHY')
    df = pd.DataFrame(train_data)
    df.columns = ['images']
    df['labels'] = train_label
    df = df.sample(frac=1).reset_index(drop=True)
    return df


# function to visualize the images
def plot(image_batch, label_batch):
    plt.figure(figsize=(10, 5))
    for i in range(10):
        ax = plt.subplot(2, 5, i + 1)
        img = cv2.imread(str(image_batch[i]))
        img = cv2.resize(img, (224, 224))
        plt.imshow(img)
        plt.title(label_batch[i])
        plt.axis("off")


# creating the classification model - data preprocessing***************************
def prepare_and_load(isval=True):
    if isval == True:
        normal_face_directory = validation_directory / 'Healthy'
        paralysis_face_directory = validation_directory / 'UnHealthy'
    else:
        normal_face_directory = test_directory / 'Healthy'
        paralysis_face_directory = test_directory / 'UnHealthy'
    normal_faces = normal_face_directory.glob('*.*')
    paralysis_faces = paralysis_face_directory.glob('*.*')
    data, labels = ([] for x in range(2))

    def prepare(case):
        for img in case:
            img = cv2.imread(str(img))
            # resizing the image to 224, 224
            img = cv2.resize(img, (224, 224))
            # some images are in gray scale 1 channel so we convert them to 3 channel
            if img.shape[2] == 1:
                img = np.dstack([img, img, img])
            # we read the images in RGB format
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # normalizing the image pixels
            img = img.astype(np.float32) / 255
            # encoding the images
            if case == normal_faces:
                label = to_categorical(0, num_classes=2)
            else:
                label = to_categorical(1, num_classes=2)
            data.append(img)
            labels.append(label)
        return data, labels

    prepare(normal_faces)
    d, l = prepare(paralysis_faces)
    d = np.array(d)
    l = np.array(l)
    return d, l


# creating the classification model - data preprocessing END***************************


# generating the data in batches for the CNN model ******************************************
# training takes place in batches, model takes th first batch, passes it through the network and
# loss is calculated at the end, then gradients travel backwards to update the parameters
def data_gen(data, batch_size):
    # getting the total number of samples in the data
    n = len(data)
    steps = n // batch_size
    # defining 2 numpy arrays for containing batch data and labels
    batch_data = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
    batch_labels = np.zeros((batch_size, 2), dtype=np.float32)

    # get the numpy array of all the indices of the input data
    indices = np.arange(n)
    # defining the counter
    i = 0
    while True:
        np.random.shuffle(indices)
        # getting the next batch
        count = 0
        next_batch = indices[(i * batch_size):(i + 1) * batch_size]
        for j, idx in enumerate(next_batch):
            img_name = data.iloc[idx]['images']
            label = data.iloc[idx]['labels']
            if label == 'Healthy':
                label = 0
            else:
                label = 1

            encoded_label = to_categorical(label, num_classes=2)
            # read the image and resize
            img = cv2.imread(str(img_name))
            img = cv2.resize(img, (224, 224))

            # check if the image is grey scale
            if img.shape[2] == 1:
                img = np.dstack([img, img, img])

            orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # normalizing the image pixels
            orig_img = img.astype(np.float32) / 255

            batch_data[count] = orig_img
            batch_labels[count] = encoded_label

            count += 1

            if count == batch_size - 1:
                break

        i == 1
        yield batch_data, batch_labels
        if i >= steps:
            i = 0


def vgg16_model(num_classes=None):
    model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))

    x = Dense(1024, activation='relu')(model.layers[-4].output)
    x = Dropout(0.7)(x)
    x = Dense(512, activation='softmax')(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation='softmax')(x)
    model = Model(model.input, x)

    return model


# creating the neural network layers
# adding a convolutional layer
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# converting 3D feature maps to 1D feature vectors
model.add(Flatten())

model.add(Dense(300))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))


batch_size = 40
nb_epochs = 1

train_data = load_train()
val_data, val_labels = prepare_and_load(isval=True)
test_data, test_labels = prepare_and_load(isval=False)

# getting the train data generator
train_data_gen = data_gen(data=train_data, batch_size=batch_size)
# getting the number of training steps
nb_train_steps = train_data.shape[0] // batch_size

# vgg_conv = vgg16_model(2)
# for layer in vgg_conv.layers[:-10]:
#     layer.trainable = False
#
# opt = Adam(lr=0.0001, decay=1e-5)
# vgg_conv.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=opt)
# history = vgg_conv.fit_generator(train_data_gen, epochs=nb_epochs, steps_per_epoch=nb_train_steps,
#                                  validation_data=(val_data, val_labels), class_weight={0: 1.0, 1: 0.4})

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# fitting the model
history = model.fit_generator(train_data_gen, epochs=nb_epochs, steps_per_epoch=nb_train_steps,
                              validation_data=(val_data, val_labels))

prediction = model.predict(test_data, batch_size=16)
prediction = np.argmax(prediction, axis=-1)
labels = np.argmax(test_labels, axis=-1)
print(classification_report(labels, prediction))

print("train data", len(train_data))
print("val_data: ", len(val_data))
print("test Data :", len(test_data))
