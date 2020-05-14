import time
st = time.time()

import os
import sys
import argparse
import pickle

import cv2
import math
import numpy as np

import shutil
import random
import json

from PIL import Image
from imutils import paths

import keras
from keras import initializers
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.preprocessing.image import img_to_array, ImageDataGenerator
#from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, Conv2DTranspose, Reshape, Flatten, Input, Dense, LeakyReLU, BatchNormalization, Dropout
#from keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Dropout
#print("COMPELTE: LOAD PACKAGE")


# Define parse
parser = argparse.ArgumentParser()
parser.add_argument("-project_dir", "--project_dir", type=str, default=None, help='Project path')
parser.add_argument("-class_dir", "--class_dir", type=str, default=None, help='Class path')
parser.add_argument("-class_list", "--class_list", type=str, default=None, help='whether use AI or not use AI')
parser.add_argument("-pid_file", "--pid_file", type=str, default='', help='PID')
args = parser.parse_args()


# PID_DIR = args.pid_dir
# if not os.path.isdir(PID_DIR):
#     os.makedirs(PID_DIR)
# pid = os.getpid()
# print("PID: ", pid, file=sys.stderr, flush=True)
# with open('generate.pid', 'wb') as file:
#     pickle.dump(pid, file)
# shutil.move(os.path.join(os.getcwd(), 'generate.pid'), os.path.join(PID_DIR, 'generate.pid'))
PID = os.getpid()
PID_FILE = args.pid_file
f = open(PID_FILE, 'w')
f.write(str(PID))
f.close()
print('PID: ', PID, file=sys.stderr, flush=True)
print('PID_FILE: ', PID_FILE, file=sys.stderr, flush=True)

# Input project_dir and status
PROJECT_DIR = args.project_dir
CLASS_DIR = args.class_dir
CLASS_LIST = args.class_list
print('PROJECT_DIR: ', PROJECT_DIR, file=sys.stderr, flush=True)
print('CLASS_DIR: ', CLASS_DIR, file=sys.stderr, flush=True)

class_list = json.loads(CLASS_LIST)
print('CLASS_LIST: ', class_list, file=sys.stderr, flush=True)





# Count epochs
ai_count = 0
no_count = 0
for one_class in class_list:
    if one_class['ai_use']:
        ai_count += 1
    else:
    	no_count += 1
total_count = ai_count * 150 + no_count * 150

# Make directory
data_dir = os.path.join(PROJECT_DIR, 'data')
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
tmp_dir = os.path.join(data_dir, 'tmp')

if not os.path.isdir(data_dir):
    os.makedirs(data_dir)
if not os.path.isdir(train_dir):
    os.makedirs(train_dir)
if not os.path.isdir(test_dir):
    os.makedirs(test_dir)
if os.path.isdir(tmp_dir):
	shutil.rmtree(tmp_dir)
if not os.path.isdir(tmp_dir):
    os.makedirs(tmp_dir)

# epoch_idx = 0
first_class = 0
present_count = 0
# Generate images
for i in range(len(class_list)):
    if class_list[i]['ai_use'] == True:
        
        max_per_class_copy = 2000
        class_name = class_list[i]['name']
#        print(class_name, ' - ai_use: true')
        
        # Make directory as temporary for 'ai_use':true
        save_path_for_downsize = os.path.join(tmp_dir, 'r48_' + class_name)
        save_path_for_increasing = os.path.join(tmp_dir, 'increasing_' + class_name)
        save_path_for_training = os.path.join(tmp_dir, 'training_' + class_name)
        save_path_for_binarization = os.path.join(tmp_dir, 'binarization_' + class_name)
        save_path_for_upsize = os.path.join(tmp_dir, 'r120_' + class_name)
        if not os.path.isdir(save_path_for_downsize):
            os.makedirs(save_path_for_downsize)
        if not os.path.isdir(save_path_for_increasing):
            os.makedirs(save_path_for_increasing)
        if not os.path.isdir(save_path_for_training):
            os.makedirs(save_path_for_training)
        if not os.path.isdir(save_path_for_binarization):
            os.makedirs(save_path_for_binarization)
        if not os.path.isdir(save_path_for_upsize):
            os.makedirs(save_path_for_upsize)
        
        # Make directory for train and test
        save_path_for_train = os.path.join(train_dir, class_name)
        save_path_for_test = os.path.join(test_dir, class_name)
        if os.path.isdir(save_path_for_train):
        	shutil.rmtree(save_path_for_train)
        if os.path.isdir(save_path_for_test):
        	shutil.rmtree(save_path_for_test)
        if not os.path.isdir(save_path_for_train):
            os.makedirs(save_path_for_train)
        if not os.path.isdir(save_path_for_test):
            os.makedirs(save_path_for_test)
        
        
        # Image class
        image_path = os.path.join(CLASS_DIR, class_name)
        image_list = list(paths.list_images(image_path))
#        print('Number of Images: ', len(image_list))
        
        # Copy Image class to train_dir
        for image in image_list:
            shutil.copy(image, save_path_for_train)
        
        
        # Resize image class as 48x48
        for path in  image_list:
            image = Image.open(path)
            resize_image = image.resize((48, 48))
            image_label = path.split((os.path.sep))[-1]
            resize_image.save(save_path_for_downsize + '/' + image_label)
            
        
        # Increasing images for training
        image_list_for_increasing = list(paths.list_images(save_path_for_downsize))
        
        train_datagen = ImageDataGenerator(rescale=1./255, 
                                           rotation_range=90,  
                                           horizontal_flip= True, 
                                           vertical_flip=True, 
                                           fill_mode='nearest')
        image_array = []
        for image in image_list_for_increasing:
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            image = img_to_array(image)
            image_array.append(image)
        image_array = np.array(image_array, dtype="float") / 128. - 1
        max_per_class_increasing = 400
        multiply_number = math.floor(max_per_class_increasing / len(image_list_for_increasing))
#        print('Multiply number: ', multiply_number)
        
        i = 0
        for batch in train_datagen.flow(image_array, 
                                        batch_size=len(image_list_for_increasing), 
                                        save_to_dir=save_path_for_increasing, 
                                        save_prefix='bw', 
                                        save_format='png'):
            i += 1
            if i > (multiply_number - 1):
                break
        # Copy images as short as 400(48x48)
        image_list_for_sub = list(paths.list_images(save_path_for_increasing))
        copy_image_list = image_list_for_increasing[:max_per_class_increasing - len(image_list_for_sub)]
        for image in copy_image_list:
            shutil.copy(image, save_path_for_increasing)
#        print('COMPLETE: IMAGE INCREASING')
        
        
        # Training DCGAN
        height = 48
        width = 48
        channels = 1
        
        image_list_for_training = list(paths.list_images(save_path_for_increasing))
        random.shuffle(image_list_for_training)
        
        train_datas = []
        for image in image_list_for_training:
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            image = img_to_array(image)
            train_datas.append(image)
        train_datas = np.array(train_datas)
        
        x_train = train_datas.reshape((train_datas.shape[0],) + (height, width, channels)).astype('float32')
        X_train = (x_train - 127.5) / 127.5
        
        # Define Model
        # latent space dimension
        latent_dim = 100
        # Image Demension
        init = initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None)
        
        # Generator network
        generator = Sequential()
        # FC:
        generator.add(Dense(144, input_shape=(latent_dim,), kernel_initializer=init))
        # FC:
        generator.add(Dense(12*12*128))
        generator.add(Reshape((12, 12, 128)))
        generator.add(Dropout(0.5))
        # Conv 1:
        generator.add(Conv2DTranspose(128, kernel_size=2, strides=2, padding='same'))
        generator.add(BatchNormalization(momentum=0.8))
        generator.add(LeakyReLU(0.2))
        # Conv 2:
        generator.add(Conv2DTranspose(128, kernel_size=2, strides=2, padding='same'))
        generator.add(BatchNormalization(momentum=0.8))
        generator.add(LeakyReLU(0.2))
        # Conv 3:
        generator.add(Conv2DTranspose(64, kernel_size=2, strides=1, padding='same'))
        generator.add(BatchNormalization(momentum=0.8))
        generator.add(LeakyReLU(0.2))
        # Conv 4:
        generator.add(Conv2DTranspose(1, kernel_size=2, strides=1, padding='same', activation='tanh'))
        
        # Discriminator network
        discriminator = Sequential()
        # Conv 1:
        discriminator.add(Conv2D(64, kernel_size=1, strides=1, padding='same', input_shape=(48, 48, 1), kernel_initializer=init))
        discriminator.add(LeakyReLU(0.2))
        # Conv 2:
        discriminator.add(Conv2D(64, kernel_size=2, strides=1, padding='same'))
        discriminator.add(BatchNormalization(momentum=0.8))
        discriminator.add(LeakyReLU(0.2))
        # Conv 3:
        discriminator.add(Conv2D(64, kernel_size=2, strides=2, padding='same'))
        discriminator.add(BatchNormalization(momentum=0.8))
        discriminator.add(LeakyReLU(0.2))
        # Conv 4:
        discriminator.add(Conv2D(64, kernel_size=2, strides=2, padding='same'))
        discriminator.add(BatchNormalization(momentum=0.8))
        discriminator.add(LeakyReLU(0.2))
        # FC
        discriminator.add(Flatten())
        discriminator.add(Dense(256))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dense(512))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.5))
        # Output
        discriminator.add(Dense(1, activation='sigmoid'))
        
        # Optimizer
        optimizer = Adam(lr=0.0002, beta_1=0.5)
        discriminator.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
        
        discriminator.trainable = False
        z = Input(shape=(latent_dim,))
        img = generator(z)
        decision = discriminator(img)
        d_g = Model(input=z, outputs=decision)
        d_g.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
        
        epochs = 150
        batch_size = 32
        smooth = 0.1
        
        real = np.ones(shape=(batch_size, 1))
        fake = np.zeros(shape=(batch_size, 1))
        
        d_loss = []
        d_g_loss = []
        
        if first_class == 0:
#                print('0 /', ai_count*150, file=sys.stdout, flush=True)
                print('0 /', total_count, file=sys.stdout, flush=True)
        for e in range(epochs):
#           print(e, file=sys.stderr, flush=True)
#            print('%d / %d'%(e+1+epoch_idx, ai_count*150), file=sys.stdout, flush=True)
            print('%d / %d'%(present_count + e + 1, total_count), file=sys.stdout, flush=True)
#            print(result_json, file=sys.stdout, flush=True)
            for i in range(len(X_train) // batch_size):
                # Train Discriminator weights
                discriminator.trainable = True
                # Real samples
                X_batch = X_train[i*batch_size:(i+1)*batch_size]
                d_loss_real = discriminator.train_on_batch(x=X_batch, y=real*(1-smooth))
                
                # Fake samples
                z = np.random.normal(loc=0, scale=1, size=(batch_size, latent_dim))
                X_fake = generator.predict_on_batch(z)
                d_loss_fake = discriminator.train_on_batch(x=X_fake, y=fake)
                
                # Discriminator loss
                d_loss_batch = 0.5 * (d_loss_real[0] + d_loss_fake[0])
                
                # Train Generator weights
                discriminator.trainable = False
                d_g_loss_batch = d_g.train_on_batch(x=z, y=real)
                
                samples = batch_size
                
                if e == (epochs-1):
                    for k in range(len(X_batch)):
                        x_fake = generator.predict(np.random.normal(loc=0, scale=1, size=(samples, latent_dim)))
                        
                        for j in range(6):
                            img1 = keras.preprocessing.image.array_to_img(x_fake[k] * 255., scale=False)
                            img1.save(os.path.join(save_path_for_training, str(e)+'_'+str(i)+'_'+str(k)+'_'+str(j)+'.png'))
            
            d_loss.append(d_loss_batch)
            d_g_loss.append(d_g_loss_batch[0])
        
        
        # binarization
        image_list_for_binarization = list(paths.list_images(save_path_for_training))
        
        for idx, img in enumerate(image_list_for_binarization):
            image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            ret, threshold_image = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
            img = Image.fromarray(threshold_image)
            img.save(os.path.join(save_path_for_binarization, 'bw_g' + str(idx) +'.png'))
#        print('COMPLETE: BINARIZATION')
            
        
        # resize 120x120
        image_list_for_upsize = list(paths.list_images(save_path_for_binarization))
        
        for path in image_list_for_upsize:
            image = Image.open(path)
            resize_image = image.resize((120, 120))
            image_label = path.split((os.path.sep))[-1]
            resize_image.save(save_path_for_upsize + '/' + image_label)
            
        
        # Copy to train_dir
        image_list_for_copy = list(paths.list_images(save_path_for_upsize))
        copy_image_list = image_list_for_copy[:max_per_class_copy - len(image_list)]
        for img in copy_image_list:
            shutil.copy(img, save_path_for_train)
        #
        move_image_list_for_test = list(paths.list_images(save_path_for_train))
        random.shuffle(move_image_list_for_test)
        for img in move_image_list_for_test[:600]:
            shutil.move(img, save_path_for_test)
#        print('COMPLETE: AI GENERATION')
        
        # Delete tmp_folder
        shutil.rmtree(tmp_dir)
        present_count += 150
    else:
        
        max_per_class = 2000
        class_name = class_list[i]['name']
#        print(class_name, ' - ai_use: false')
        
        
        # Make directory as temporary for 'ai_use':false
        save_path_for_increasing = os.path.join(tmp_dir, 'increasing_' + class_name)
        if not os.path.isdir(save_path_for_increasing):
            os.makedirs(save_path_for_increasing)
        
        # Make directory for train and test
        save_path_for_train = os.path.join(train_dir, class_name)
        save_path_for_test = os.path.join(test_dir, class_name)
        if os.path.isdir(save_path_for_train):
        	shutil.rmtree(save_path_for_train)
        if os.path.isdir(save_path_for_test):
        	shutil.rmtree(save_path_for_test)
        if not os.path.isdir(save_path_for_train):
            os.makedirs(save_path_for_train)
        if not os.path.isdir(save_path_for_test):
            os.makedirs(save_path_for_test)
        
        
        # Image class
        image_path = os.path.join(CLASS_DIR, class_name)
        image_list = list(paths.list_images(image_path))
#        print('Number of Images: ', len(image_list))
        
        # Copy Image class to train_dir
        for image in image_list:
            shutil.copy(image, save_path_for_train)
        
        
        # Generate images
        train_datagen = ImageDataGenerator(rescale=1./255, 
                                           rotation_range=90, 
                                           horizontal_flip=True, 
                                           vertical_flip=True, 
                                           fill_mode='nearest')
        image_array = []
        for image in image_list:
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            image = img_to_array(image)
            image_array.append(image)
        image_array = np.array(image_array, dtype="float") / 128. - 1
        
        
        multiply_number = math.ceil(max_per_class / len(image_list))
#        print('Multiply number: ', multiply_number)
        
        i = 0
        for batch in train_datagen.flow(image_array, 
                                        batch_size = len(image_list), 
                                        save_to_dir = save_path_for_increasing, 
                                        save_prefix='bw', 
                                        save_format='png'):
            i += 1
            if i > (multiply_number-1):
                break
                
        # Copy generated images to train_dir
        image_list_for_copy = list(paths.list_images(save_path_for_increasing))
        copy_image_list = image_list_for_copy[:max_per_class - len(image_list)]
        for image in copy_image_list:
            shutil.copy(image, save_path_for_train)
        

        if first_class == 0:
                print('0 /', total_count, file=sys.stdout, flush=True)

        # Move generated images to test_dir
        move_image_list = list(paths.list_images(save_path_for_train))
        random.shuffle(move_image_list)
        for idx, image in enumerate(move_image_list[:600]):
        	shutil.move(image, save_path_for_test)
        	if (idx+1) % 4 == 0:
        		print('%d / %d'%(present_count+(idx//4)+1, total_count), file=sys.stdout, flush=True)
            # shutil.move(image, save_path_for_test)
#        print('COMPLETE: SIMPLE GENERATION')

        # Delete tmp_dir
        shutil.rmtree(tmp_dir)
        present_count += 150
#    epoch_idx += 150
    first_class += 1
ed = time.time()
full_time = ed - st
print("full_time: ", full_time)