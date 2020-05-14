import time
st = time.time()

import os
import sys
import shutil
import argparse
import numpy as np

import keras
from keras import Model, backend, Sequential
from keras.models import save_model, load_model

from keras.layers import Dense
from keras.activations import relu

from keras.optimizers import RMSprop, SGD
from keras.callbacks import CSVLogger

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet_v2 import MobileNetV2

# Convert tensorflow model
import tensorflow as tf
import tensorflow.python.saved_model
# The export path contatins the name and the version of the model
tf.keras.backend.set_learning_phase(0) # ignore dropout at inference

config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction=0.4
session = tf.Session(config=config)


# Define parse
parser = argparse.ArgumentParser()
parser.add_argument("-project_dir", "--project_dir", type=str, default=None, help='Project path')
parser.add_argument("-epochs", "--epochs", type=int, default=50, help='Input epochs for trainig')
parser.add_argument("-batch_size", "--batch_size", type=int, default=16, help='Input batch_size considering memory size')
parser.add_argument("-learning_rate", "--learning_rate", type=float, default=0.001, help='Input the learning_rate to use for the optimization function')
parser.add_argument("-pid_file", "--pid_file", type=str, default='', help='PID')
args = parser.parse_args()


PID = os.getpid()
PID_FILE = args.pid_file
f = open(PID_FILE, 'w')
f.write(str(PID))
f.close()
print('PID: ', PID, file=sys.stderr, flush=True)
print('PID_FILE: ', PID_FILE, file=sys.stderr, flush=True)


# Input PROJECT_DIR, EPOCHS, BATCH_SIZE and LEARNING_RATE
PROJECT_DIR = args.project_dir
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate

train_dir = os.path.join(PROJECT_DIR, 'data', 'train')
test_dir = os.path.join(PROJECT_DIR, 'data', 'test')
label_list = list(os.listdir(train_dir))
NUM_OF_CLASS = int(len(label_list))
#print('NUM_OF_CLASS: ', NUM_OF_CLASS)

# if NUM_OF_CLASS == 2:
#     LEARNING_RATE = LEARNING_RATE / 10
# else:
#     LEARNING_RATE = LEARNING_RATE

print("PROJECT_DIR: ", PROJECT_DIR, file=sys.stderr, flush=True)
print("EPOCHS: ", EPOCHS, file=sys.stderr, flush=True)
print("BATCH_SIZE: ", BATCH_SIZE, file=sys.stderr, flush=True)
print("LEARNING_RATE: ", LEARNING_RATE, file=sys.stderr, flush=True)
print("NUM_OF_CLASS: ", NUM_OF_CLASS, file=sys.stderr, flush=True)
PID_FILE = args.pid_file
print('PID_FILE: ', PID_FILE, file=sys.stderr, flush=True)

# Model name
MODEL_NAME = PROJECT_DIR.split('/')[-1]
#print("MODEL_NAME: ", MODEL_NAME, file=sys.stderr, flush=True)


# Delete model_dir and training.log
if os.path.isdir(os.path.join(PROJECT_DIR, 'model')):
	shutil.rmtree(os.path.join(PROJECT_DIR, 'model'))
	print("DELETE: ", os.path.join(PROJECT_DIR, 'model'), file=sys.stderr, flush=True)
if os.path.isdir(os.path.join(PROJECT_DIR, 'run')):
	shutil.rmtree(os.path.join(PROJECT_DIR, 'run'))
	print("DELETE: ", os.path.join(PROJECT_DIR, 'run'), file=sys.stderr, flush=True)


# Make directory
model_dir = os.path.join(PROJECT_DIR, 'model')
run_dir = os.path.join(PROJECT_DIR, 'run')
tmp_dir = os.path.join(PROJECT_DIR, 'tmp')
if not os.path.isdir(run_dir):
    os.makedirs(run_dir)
    print("CREATE: ", run_dir, file=sys.stderr, flush=True)
if not os.path.isdir(tmp_dir):
    os.makedirs(tmp_dir)
    print("CREATE: ", tmp_dir, file=sys.stderr, flush=True)


train_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(
	train_dir, 
	target_size=(224, 224),
    class_mode='categorical', 
    batch_size=BATCH_SIZE)
test_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(
	test_dir, 
	target_size=(224, 224),
    class_mode='categorical', 
    batch_size=BATCH_SIZE)


# Define network
mobilenet = keras.applications.mobilenet_v2.MobileNetV2(backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
x = mobilenet.layers[-2].output
x = Dense(128, activation='relu')(x)

csv_logger = CSVLogger(os.path.join(run_dir, 'training.log'))


if NUM_OF_CLASS == 2:
    predictions = Dense(NUM_OF_CLASS, activation='sigmoid')(x)
    model = Model(inputs= mobilenet.input, outputs=predictions)
    model.compile(RMSprop(lr=LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
else:
    predictions = Dense(NUM_OF_CLASS, activation='softmax')(x)
    model = Model(inputs= mobilenet.input, outputs=predictions)
    model.compile(SGD(lr=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit_generator(train_batches, steps_per_epoch=10, validation_data=test_batches, validation_steps=10, epochs=EPOCHS, verbose=2, callbacks=[csv_logger])
print("COMPLETE: TRAINING", file=sys.stderr, flush=True)


# Save model
save_model(model, os.path.join(tmp_dir, MODEL_NAME + '.h5'))
print("COMPLETE: SAVE MODEL", file=sys.stderr, flush=True)


# Load and Convert model
model = load_model(os.path.join(tmp_dir, MODEL_NAME + '.h5'))

with keras.backend.get_session() as sess:
	tf.saved_model.simple_save(
		sess,
		model_dir, 
		inputs={'input_image': model.input}, 
		outputs={t.name: t for t in model.outputs})


# Move model to '1'
move_dir = os.path.join(PROJECT_DIR, 'model', '1')
if not os.path.isdir(move_dir):
	os.makedirs(move_dir)
	# Move, Copy, Remove
	shutil.move(os.path.join(model_dir, 'saved_model.pb'), os.path.join(move_dir, 'saved_model.pb'))
	shutil.copytree(os.path.join(model_dir, 'variables'), os.path.join(move_dir, 'variables'))
	shutil.rmtree(os.path.join(model_dir, 'variables'))
	print("COMPLETE: MOVE", file=sys.stderr, flush=True)


# Delete model(.h5)
if os.path.isdir(tmp_dir):
	shutil.rmtree(tmp_dir)
	print("DELETE: ", tmp_dir, file=sys.stderr, flush=True)


ed = time.time()
full_time = ed - st
print("full_time: ", full_time, file=sys.stderr, flush=True)