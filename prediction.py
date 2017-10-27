import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.models import load_model
import pandas as pd

import io
import bson
import json
import cv2
import matplotlib.pyplot as plt
import concurrent.futures
from multiprocessing import cpu_count
from datetime import datetime

num_images_test = 10000
num_image_total = 176
#num_image_batch = 100
im_size = 180
num_cpus = cpu_count()
num_classes = 5270  # This will reduce the max accuracy to about 0.75

model = load_model('epoch5top1000.h5')

print("Model load completed")
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

f = open('/datadrive/Cdiscount/test.bson', 'rb')
bson_data_iter = bson.decode_file_iter(f)

with open("id2category.json", "r") as id2category_file:
    id2category = json.load(id2category_file)

submission = pd.read_csv('./sample_submission.csv', index_col='_id')

most_frequent_guess = 1000018296
submission['category_id'] = most_frequent_guess # Most frequent guess

def load_next_batch():
    with concurrent.futures.ThreadPoolExecutor(num_cpus) as executor:
        future_load = []
    
        for i,d in enumerate(bson_data_iter):
            if i >= num_images_test:
                break
            future_load.append(executor.submit(load_image, d['imgs'][0]['picture'], d['_id']))

        print("Starting future processing")
        x_array = np.empty((num_images_test, im_size, im_size, 3), dtype=np.float32)
        id_array = []
        for i, future in enumerate(concurrent.futures.as_completed(future_load)):
            x, _id = future.result()
            x_array[i] = x
            id_array.append(_id)
        index_array = model.predict(x_array)
        print("index_array shape is:", np.shape(index_array))
        index_array = np.argmax(index_array, axis=1)
        print("index_array shape is:", np.shape(index_array))
        for idx, v in enumerate(index_array):
            submission.loc[id_array[idx], 'category_id'] = id2category[str(v)]
        
    
#        y_cat = np.argmax(model.predict(x[None])[0])
#        y_cat = id2category[str(y_cat)]
        
#            submission.loc[_id, 'category_id'] = y_cat
            

for i in range(num_image_total):
        print("========= Start batch %s ===========" % i)
        start_time = datetime.now()
        load_next_batch()
        end_time = datetime.now()
        if i%10 == 0:
            submission.to_csv("new_submission_%s.csv.gz" % i, compression='gzip')
        print("start_time for batch %s is %s and end_time is %s" % (i, start_time, end_time))

submission.to_csv('new_submission.csv.gz', compression='gzip')
