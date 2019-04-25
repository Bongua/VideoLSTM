from keras.applications.vgg16 import VGG16
from keras.models import Model,Sequential
from keras.layers import Dense, Input,Conv2D,MaxPooling2D,Flatten,Dropout
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Nadam
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.sequence import TimeseriesGenerator
import cv2
import os
import numpy as np

model = Sequential()
model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2), activation='relu', padding='same'),
                          input_shape=(5, 256, 256, 3)))
model.add(TimeDistributed(Conv2D(32, (3, 3), kernel_initializer="he_normal", activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
model.add(Dropout(0.2))
model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
model.add(Dropout(0.2))
model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu')))
model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
model.add(Dropout(0.2))
model.add(TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu')))
model.add(TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
model.add(Dropout(0.2))
model.add(TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu')))
model.add(TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))


model.add(TimeDistributed(Flatten()))

model.add(Dropout(0.5))

model.add(LSTM(256, return_sequences=False, dropout=0.5))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
model.summary()


def data_generator(path, size=(224, 224),seq_len=0, batch_size=1):
    st = 0
    curr_dir = 0
    dirs = [dir for dir in os.listdir(path)]
    images_name = [os.listdir(path + dir_) for dir_ in dirs]
    images_r_d = []

    while True:
        curr_b = 0
        y_batch = np.zeros((batch_size, len(dirs)))
        x_batch = np.zeros((batch_size,seq_len,*size,3))
        while True:
            names = images_name[curr_dir]

            if len(images_r_d)==len(dirs):
                images_r_d[curr_dir].pop(0)
                curr_im = cv2.imread(path+dirs[curr_dir]+"\\"+names[st+seq_len])
                curr_im =cv2.resize(curr_im,size)

                images_r_d[curr_dir].append(curr_im)
                x_batch[curr_b] = images_r_d[curr_dir]
                y_batch[curr_b, curr_dir] = 1.0

            else:

                images_r = [cv2.imread(path+dirs[curr_dir]+"\\"+name) for name in names[st:st+seq_len] ]
                images_r = [cv2.resize(im,size) for im in images_r ]
                print(np.array(images_r).shape)
                images_r_d.append(images_r)
                x_batch[curr_b] = images_r
                y_batch[curr_b,curr_dir] = 1.0

            curr_b +=1
            curr_dir+=1
            if curr_dir == len(dirs):

                curr_dir = 0
                if (st+1 == (len(names) - seq_len)):
                    st = 0
                else:
                    st += 1

            if curr_b == batch_size:break

        x_batch/=255.

        yield (x_batch, y_batch)





model.fit_generator(
    data_generator("D:\\Tensors\\CNN-Data\\", size=(256, 256), seq_len=5, batch_size=10),
        steps_per_epoch=30,
        epochs=2,

        )






cv2.waitKey(0)

model.save()
cap = cv2.VideoCapture("адажио из балета Щелкунчик.mp4")
frame_c = 0
seq = 0
fr = []
while True:
    ret,frame = cap.read()
    frame = cv2.resize(frame,(256,256))
    cv2.imshow("capture",frame)
    cv2.waitKey(1)
    if not ret:break

    fr.append(frame)
    if len(fr) == 5:
        frames = np.array(fr,float).reshape((1,5,256,256,3))
        frames/=255.
        print(model.predict(frames))
        seq = 0
        fr.pop(0)
    frame_c+=1
