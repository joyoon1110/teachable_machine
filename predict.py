import time
st = time.time()

import os
import sys
import argparse
import json

import numpy as np
import requests
#from keras.applications import inception_v3
from keras.preprocessing import image


# Argument parser for giving input image_path from command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_path", required=True, help="path of the image")
ap.add_argument("-project_dir", "--project_dir", type=str, default=None, help='Project path')
ap.add_argument("-port_num", "--port_num", type=int, default=None, help='Input rest_api_port number')
ap.add_argument("-model_name", "--model_name", type=str, default=None, help='Please write model name')
args = vars(ap.parse_args())

image_path = args['image_path']
# Preprocessing our input image
img = image.img_to_array(image.load_img(image_path, target_size=(224, 224))) / 255.


# this line is added because of a bug in tf_serving(1.10.0-dev)
img = img.astype('float64')

payload = {
    "instances": [{'input_image': img.tolist()}]
}

PORT_NUM = str(args['port_num'])
MODEL_NAME = args['model_name']
RP = 'http://localhost:' + PORT_NUM + '/v1/models/' + MODEL_NAME + ':predict'
#print('RP: ', RP)

# sending post request to TensorFlow Serving server
r = requests.post(RP, json=payload)
pred = json.loads(r.content.decode('utf-8'))


PROJECT_DIR = args['project_dir']
train_dir = os.path.join(PROJECT_DIR, 'data', 'train')
label_list = list(os.listdir(train_dir))
#NUM_OF_CLASS = int(len(label_list))

all_class_matching = []
for idx, label in enumerate(label_list):
	class_matching = {}
	class_matching['name'] = label
	class_matching['value'] = int(round(pred['predictions'][0][idx], 2) * 100)
	all_class_matching.append(class_matching)

#print('list_class_matching: ', all_class_matching)
print(all_class_matching, file=sys.stdout, flush=True)

# ed = time.time()
# print("Full time: ", ed - st)