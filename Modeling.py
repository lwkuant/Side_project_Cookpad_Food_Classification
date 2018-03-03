# -*- coding: utf-8 -*-
"""
JSAI Cup 2018
https://deepanalytics.jp/compe/59?tab=comperank
"""

### Initial Setup
import numpy as np
import pandas as pd 
import os
import re
from time import time
seed = 1000
from collections import Counter
import shutil
import pickle
from skimage.io import imshow

os.chdir('D:/Google雲端硬碟/Project/Competition_JSAI_Cup_2018/Datasets')


### Allocate the data into Training and Validation Set
df = pd.read_csv('train_master.tsv', sep = '\t')

from sklearn.model_selection import train_test_split
df_train, df_val = train_test_split(df.ix[8000:, ], train_size = 2000, random_state = seed,
                                     shuffle = True, stratify = df.ix[8000:, ]['category_id'])

class_label = sorted(list(set(df['category_id'])))
'''
base_dir = 'D:/Google雲端硬碟/Project/Competition_JSAI_Cup_2018/Datasets/data_train'
for label in class_label:
    os.mkdir(os.path.join(base_dir, str(label)))

for ind, file_name in enumerate(df['file_name'].values[:8000]):
    shutil.copy(os.path.join(base_dir, file_name),
                    os.path.join(base_dir, str(df['category_id'].values[:8000][ind])))


base_dir = 'D:/Google雲端硬碟/Project/Competition_JSAI_Cup_2018/Datasets/data_val'
for label in class_label:
    os.mkdir(os.path.join(base_dir, str(label)))

for ind, file_name in enumerate(df_val['file_name'].values):
    print(ind)
    shutil.copy(os.path.join(base_dir, file_name),
                    os.path.join(base_dir, str(df_val['category_id'].values[ind])))


for ind, file_name in enumerate(df_train['file_name'].values):
    print(ind)
    shutil.copy(os.path.join('D:/Google雲端硬碟/Project/Competition_JSAI_Cup_2018/Datasets/data_val', file_name),
                    os.path.join('D:/Google雲端硬碟/Project/Competition_JSAI_Cup_2018/Datasets/data_train', str(df_train['category_id'].values[ind])))
'''

### Set up the Important Variables
pic_scale_down_factor = 4
height = int(768/pic_scale_down_factor)
width = int(1024/pic_scale_down_factor)
num_class = 55
num_train = 10000
num_val = 1995
train_dir = 'D:/Google雲端硬碟/Project/Competition_JSAI_Cup_2018/Datasets/data_train'
val_dir = 'D:/Google雲端硬碟/Project/Competition_JSAI_Cup_2018/Datasets/data_val'


###
### Pretrained Model (VGG19)
###
batch_size = 4
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG19

conv_base = VGG19(weights='imagenet',
                  include_top=False,
                  input_shape=(height, width, 3))

conv_base.summary() # shape of last layer = (6, 8, 512) 
final_layer_shape = [6, 8, 512]

datagen = ImageDataGenerator(rescale=1./255)

def extract_features(directory, sample_count, batch_size):
    features = np.zeros(shape=([sample_count] + final_layer_shape))
    labels = np.zeros(shape=(sample_count, num_class))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(height, width),
        batch_size=batch_size,
        class_mode='categorical')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        print(i)
        if i * batch_size >= sample_count:
            break
    return features, labels

start_time = time()
train_features, train_labels = extract_features(train_dir, num_train, batch_size) # 
print(time() - start_time)

start_time = time()
validation_features, validation_labels = extract_features(val_dir, num_val, 1) # Since the number of validation set cannot be divided completely by 4
print(time() - start_time)

train_features = np.reshape(train_features, (num_train, np.prod(final_layer_shape)))
validation_features = np.reshape(validation_features, (num_val, np.prod(final_layer_shape)))

#with open('./train_features', 'wb') as fp:
#    pickle.dump(train_features, fp)
#with open('./train_labels', 'wb') as fp:
#    pickle.dump(train_labels, fp)
#with open('./validation_features', 'wb') as fp:
#    pickle.dump(validation_features, fp)
#with open('./validation_labels', 'wb') as fp:
#    pickle.dump(validation_labels, fp)    

with open('./train_features', 'rb') as fp:
    train_features = pickle.load(fp)
with open('./train_labels', 'rb') as fp:
    train_labels = pickle.load(fp)
with open('./validation_features', 'rb') as fp:
    validation_features = pickle.load(fp)
with open('./validation_labels', 'rb') as fp:
    validation_labels = pickle.load(fp)    


from keras import models
from keras import layers
from keras import optimizers

np.random.seed(seed)
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=np.prod(final_layer_shape)))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_class, activation='softmax'))
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
    loss='categorical_crossentropy',
    metrics=['acc'])
history = model.fit(train_features, train_labels,
    epochs=50,
    batch_size=20,
    validation_data=(validation_features, validation_labels))

import matplotlib.pyplot as plt
import seaborn as sns
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

sns.set_style("darkgrid")
plt.figure(figsize=[20, 15])
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.plot(epochs, acc, 'ko', label='Training Accuracy')
plt.plot(epochs, val_acc, 'k', label='Validation Accuracy')
plt.yticks(np.linspace(0, 1, 6))
plt.xticks(np.linspace(0, 50, 6))
plt.title('Training and Validation Accuracy', fontsize=25)
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.legend(loc=4, prop={'size': 15})
plt.savefig('Accuracy_VGG19_1_layer_256.png')

sns.set_style("darkgrid")
plt.figure(figsize=[20, 15])
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.plot(epochs, loss, 'ko', label='Training Loss')
plt.plot(epochs, val_loss, 'k', label='Validation Loss')
plt.title('Training and Validation Loss', fontsize=25)
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.legend(loc=0, prop={'size': 15})
plt.savefig('Loss_VGG19_1_layer_256.png')

###
### Pretrained Model (VGG19) with Data Augmentation
###

batch_size = 1
times_augmentation = 4

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG19

conv_base = VGG19(weights='imagenet',
                  include_top=False,
                  input_shape=(height, width, 3))

conv_base.summary() # shape of last layer = (6, 8, 512) 
final_layer_shape = [6, 8, 512]

'''
datagen = ImageDataGenerator(rescale=1./255)

datagen_augmentation = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

def extract_features_with_augmentation(directory, sample_count, batch_size, times_augmentation):
    features = np.zeros(shape=([sample_count*times_augmentation] + final_layer_shape))
    labels = np.zeros(shape=(sample_count*times_augmentation, num_class))
    
    generator = datagen.flow_from_directory(
        directory,
        target_size=(height, width),
        batch_size=batch_size,
        class_mode='categorical')
    
    i = 0

    for inputs_batch, labels_batch in generator:
        j = 0
        for batch in datagen_augmentation.flow(inputs_batch, batch_size = batch_size):
            
            features_batch = conv_base.predict(batch)
            features[i*times_augmentation + j] = features_batch
            labels[i*times_augmentation + j] = labels_batch
            j += 1
            if j % 4 == 0:
                break
        i += 1
        print(i)
        if i >= sample_count:
            break

    return features, labels



start_time = time()
train_features, train_labels = extract_features_with_augmentation(train_dir, num_train, batch_size,
                                                times_augmentation) # 6035.708188533783s
print(time() - start_time)
'''

#with open('./train_features_augmentation_40k', 'wb') as fp:
#    pickle.dump(train_features, fp, protocol=4)
#with open('./train_labels_augmentation_40k', 'wb') as fp:
#    pickle.dump(train_labels, fp)

with open('./train_features_augmentation_40k', 'rb') as fp:
    train_features = pickle.load(fp)
with open('./train_labels_augmentation_40k', 'rb') as fp:
    train_labels = pickle.load(fp)

with open('./validation_features', 'rb') as fp:
    validation_features = pickle.load(fp)
with open('./validation_labels', 'rb') as fp:
    validation_labels = pickle.load(fp)    
    
np.random.seed(seed)
augmentation_ind = np.random.choice(np.arange(num_train*times_augmentation),
                                    num_train*times_augmentation, replace = False)

train_features = train_features[augmentation_ind]
train_labels = train_labels[augmentation_ind]

train_features = np.reshape(train_features, (num_train*times_augmentation, np.prod(final_layer_shape)))
validation_features = np.reshape(validation_features, (num_val, np.prod(final_layer_shape)))


from keras import models
from keras import layers
from keras import optimizers

np.random.seed(seed)
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=np.prod(final_layer_shape)))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_class, activation='softmax'))
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
    loss='categorical_crossentropy',
    metrics=['acc'])
history = model.fit(train_features, train_labels,
    epochs=50,
    batch_size=20,
    validation_data=(validation_features, validation_labels))


import matplotlib.pyplot as plt
import seaborn as sns
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

sns.set_style("darkgrid")
plt.figure(figsize=[20, 15])
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.plot(epochs, acc, 'ko', label='Training Accuracy')
plt.plot(epochs, val_acc, 'k', label='Validation Accuracy')
plt.yticks(np.linspace(0, 1, 6))
plt.xticks(np.linspace(0, 50, 6))
plt.title('Training and Validation Accuracy', fontsize=25)
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.legend(loc=4, prop={'size': 15})
plt.savefig('Accuracy_VGG19_Data_Augmentation_40k_1_layer_256.png')

sns.set_style("darkgrid")
plt.figure(figsize=[20, 15])
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.plot(epochs, loss, 'ko', label='Training Loss')
plt.plot(epochs, val_loss, 'k', label='Validation Loss')
plt.title('Training and Validation Loss', fontsize=25)
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.legend(loc=0, prop={'size': 15})
plt.savefig('Loss_VGG19_Data_Augmentation_40k_1_layer_256.png')


###
### Pretrained Model (VGG19) with Retraining the Block5 in VGG19
###

###
batch_size = 1
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG19

conv_base = VGG19(weights='imagenet',
                  include_top=False,
                  input_shape=(height, width, 3))

conv_base_raw = VGG19(weights='imagenet',
                  include_top=False,
                  input_shape=(height, width, 3))

for i in range(5):
    conv_base.layers.pop()

conv_base.outputs = [conv_base.layers[-1].output]
conv_base.layers[-1].outbound_nodes = []

conv_base.summary() # shape of last layer = (12, 16, 512) 
final_layer_shape = [12, 16, 512]

datagen = ImageDataGenerator(rescale=1./255)

def extract_features(directory, sample_count, batch_size):
    features = np.zeros(shape=([sample_count] + final_layer_shape))
    labels = np.zeros(shape=(sample_count, num_class))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(height, width),
        batch_size=batch_size,
        class_mode='categorical')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        print(i)
        if i * batch_size >= sample_count:
            break
    return features, labels

start_time = time()
train_features, train_labels = extract_features(train_dir, num_train, batch_size) # 1289.7562346458435s
print(time() - start_time)

start_time = time()
validation_features, validation_labels = extract_features(val_dir, num_val, 1) # 257.6586627960205s
print(time() - start_time)

#train_features = np.reshape(train_features, (num_train, np.prod(final_layer_shape)))
#validation_features = np.reshape(validation_features, (num_val, np.prod(final_layer_shape)))

#with open('./train_features_without_block5', 'wb') as fp:
#    pickle.dump(train_features, fp, protocol = 4)
#with open('./train_labels_without_block5', 'wb') as fp:
#    pickle.dump(train_labels, fp)
#with open('./validation_features_without_block5', 'wb') as fp:
#    pickle.dump(validation_features, fp)
#with open('./validation_labels_without_block5', 'wb') as fp:
#    pickle.dump(validation_labels, fp)    
    
with open('./train_features_without_block5', 'rb') as fp:
    train_features = pickle.load(fp)
with open('./train_labels_without_block5', 'rb') as fp:
    train_labels = pickle.load(fp)
with open('./validation_features_without_block5', 'rb') as fp:
    validation_features = pickle.load(fp)
with open('./validation_labels_without_block5', 'rb') as fp:
    validation_labels = pickle.load(fp)    


from keras import models
from keras import layers
from keras import optimizers

#from keras.callbacks import EarlyStopping
#early_stopping_monitor = EarlyStopping(monitor = 'acc', patience=5)

np.random.seed(seed)
model = models.Sequential()
model.add(layers.ZeroPadding2D((1,1), input_shape=(12, 16, 512)))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2), strides=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu', input_dim=np.prod(final_layer_shape)))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_class, activation='softmax'))
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
    loss='categorical_crossentropy',
    metrics=['acc'])
history = model.fit(train_features, train_labels,
    epochs=30,
    batch_size=20,
    validation_data=(validation_features, validation_labels))

import matplotlib.pyplot as plt
import seaborn as sns
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

sns.set_style("darkgrid")
plt.figure(figsize=[20, 15])
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.plot(epochs, acc, 'ko', label='Training Accuracy')
plt.plot(epochs, val_acc, 'k', label='Validation Accuracy')
plt.yticks(np.linspace(0, 1, 6))
plt.xticks(np.linspace(0, 30, 4))
plt.title('Training and Validation Accuracy', fontsize=25)
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.legend(loc=4, prop={'size': 15})
plt.savefig('Accuracy_VGG19_Retraining_Block5_1_layer_256.png')

sns.set_style("darkgrid")
plt.figure(figsize=[20, 15])
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.plot(epochs, loss, 'ko', label='Training Loss')
plt.plot(epochs, val_loss, 'k', label='Validation Loss')
plt.xticks(np.linspace(0, 30, 4))
plt.title('Training and Validation Loss', fontsize=25)
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.legend(loc=0, prop={'size': 15})
plt.savefig('Loss_VGG19_Retraining_Block5_1_layer_256.png')