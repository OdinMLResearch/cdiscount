import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import pandas as pd

import io
import bson
import json
import cv2
import matplotlib.pyplot as plt
import concurrent.futures
from multiprocessing import cpu_count
from datetime import datetime

num_image_batch = 40000
num_image_total = 176
#num_image_batch = 100
im_size = 180
num_cpus = cpu_count()
num_classes = 5270  # This will reduce the max accuracy to about 0.75


model = load_model('fix_inception.h5')
print("Model load completed")
model.summary()

for layer in model.layers:
    layer.trainable = True

print("Make Model trainable")
model.summary()


def imread(buf):
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_ANYCOLOR)

def img2feat(im):
    x = cv2.resize(im, (im_size, im_size), interpolation=cv2.INTER_AREA)
    return np.float32(x) / 255

def load_image(pic, target):
    picture = imread(pic)
    x = img2feat(picture)
    
    return x, target

f = open('/datadrive/Cdiscount/train.bson', 'rb')
bson_data_iter = bson.decode_file_iter(f)

with open("category2id.json", "r") as category2id_file:
    category2id = json.load(category2id_file)

def load_next_batch():
        print("Loading Next Batch ....")
        X = np.empty((num_image_batch, im_size, im_size, 3), dtype=np.float32)
        y = []
        with concurrent.futures.ThreadPoolExecutor(num_cpus) as executor:

            delayed_load = []

            i = 0
            try:
                for c, d in enumerate(bson_data_iter):
                    target = d['category_id']
                    pic = d['imgs'][0]
                    delayed_load.append(executor.submit(load_image, pic['picture'], target))
                        
                    i = i + 1

                    if i >= num_image_batch:
                        raise IndexError()

            except IndexError:
                pass;
            
            for i, future in enumerate(concurrent.futures.as_completed(delayed_load)):
                x, target = future.result()
                
                X[i] = x
                y.append(target)

        print("Shape of X and y\n")
        print(X.shape, len(y))

        y = pd.Series(y)
        y = [category2id[str(c)] for c in y]

        #print("Y is: %s" % y)

        return X, y

# Train a simple NN

#model = Sequential()
#model.add(Conv2D(32, 3, activation='relu', padding='same', input_shape=X.shape[1:]))
#model.add(Conv2D(32, 3, activation='relu', padding='same', input_shape=X.shape[1:]))
#model.add(MaxPooling2D(2))
#
#model.add(Dropout(0.1))
#model.add(Conv2D(64, 3, activation='relu', padding='same'))
#model.add(Conv2D(64, 3, activation='relu', padding='same'))
#model.add(MaxPooling2D(2))
#
#model.add(Conv2D(128, 3, activation='relu', padding='same'))
#model.add(Conv2D(128, 3, activation='relu', padding='same'))
#model.add(Conv2D(128, 3, activation='relu', padding='same'))
#model.add(MaxPooling2D(2))
#
#model.add(Conv2D(512, 3, activation='relu', padding='same'))
#model.add(Conv2D(512, 3, activation='relu', padding='same'))
#model.add(Conv2D(512, 3, activation='relu', padding='same'))
#model.add(Conv2D(512, 3, activation='relu', padding='same'))
#model.add(MaxPooling2D(2))
#
#model.add(Flatten())
#model.add(Dense(4096, activation='relu'))
#model.add(Dropout(0.4))
#model.add(Dense(4096, activation='relu'))
#model.add(Dropout(0.4))
#model.add(Dense(num_classes, activation='softmax'))


model.compile('rmsprop', 'sparse_categorical_crossentropy', metrics=['accuracy'])

for i in range(num_image_total):
        print("========= Start batch %s ===========" % i)
        X, y = load_next_batch()
        start_time = datetime.now()
        model.fit(X, y, batch_size=64, validation_split=0.1, epochs=1, shuffle=True)
        #loss = model.train_on_batch(X, y)
        end_time = datetime.now()
        print("start_time for batch %s is %s and end_time is %s" % (i, start_time, end_time))
        if i%10 == 0 :
            filename = "/datadrive/result/inceptionv3_%s.ht" % i
            model.save(filename)

filename = "/datadrive/result/inceptionv3.h5"
print("Saving model to local file:", filename)
model.save(filename)
