import numpy as np 
import pickle
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def make_model():

    model = Sequential()

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(UpSampling2D((2, 2)))
    
    model.add(Conv2DTranspose(2, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))    

    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(0.001))

    return model


## ARCHITECTURE ADAPTED FROM https://github.com/zhixuhao/unet ##
def unet(input_size):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    # conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(2, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 4e-5), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # model.summary()

    return model


with open("x_train.pickle", "rb") as input_file:
    x_train = pickle.load(input_file)

with open("y_train.pickle", "rb") as input_file:
    y_train = pickle.load(input_file)

with open("x_test.pickle", "rb") as input_file:
    x_test = pickle.load(input_file)

with open("y_test.pickle", "rb") as input_file:
    y_test = pickle.load(input_file)

model = unet(x_train.shape[1:])

model.fit(x_train, y_train, nb_epoch=50 ,batch_size=32, shuffle=True, validation_split = 0.1)
model.save("sunet.h5")

from keras.models import load_model
model = load_model('sunet.h5')

y_pred = model.predict(x_test)

with open("y_test.pickle", "rb") as input_file:
    y_test = pickle.load(input_file)

# calculate pixel accuracy 
acc = 0.0
y_pred = np.round(y_pred)
y_pred = y_pred.astype(int)
y_test = y_test.astype(int)
for i in range(y_pred.shape[0]):
  acc += np.count_nonzero(y_pred[i] == y_test[i])/(y_pred[i].size)
print(acc/y_pred.shape[0])

def iou(y_true, y_pred):
  y_true = y_true.astype(bool)
  y_pred = y_pred.astype(bool)
  overlap = y_true*y_pred # Logical AND
  union = y_true + y_pred # Logical OR
  IOU = overlap.sum()/float(union.sum())

  return IOU 

print(iou(y_test, y_pred))

from keras import backend as K
def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)

  iou = K.eval(iou)
  return iou

print(iou_coef(y_test, y_pred))

def dice_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  dice = K.mean((2 * intersection + smooth)/(union + smooth), axis=0)

  dice = K.eval(dice)
  return dice

print(dice_coef(y_test, y_pred))

y_n_pred = np.argmin(y_pred, axis=-1)
y_n_test = np.argmin(y_test, axis=-1)
acc = 0.0

for i in range(y_n_pred.shape[0]):
  acc += np.count_nonzero(y_n_pred[i] == y_n_test[i])/(y_n_pred[i].size)
print("acc=", acc/y_n_pred.shape[0])

for i in range(10, 60, 10):
  img = y_n_pred[i]
  new_img = y_n_test[i]
  print(y_n_pred[i].shape)
  fname = "unet2_" + str(i) +".png"
  plt.imsave(fname, img, cmap=cm.gray)