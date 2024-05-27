import os
import random
import shutil
from shutil import copyfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.optimizers import RMSprop

root_dir = './data/'

if os.path.exists(root_dir):
    shutil.rmtree(root_dir)

def create_train_val_dirs(root_dir):
    os.makedirs(os.path.join(root_dir, 'training/cheetahs'))
    os.makedirs(os.path.join(root_dir, 'training/lions'))
    os.makedirs(os.path.join(root_dir, 'validation/cheetahs'))
    os.makedirs(os.path.join(root_dir, 'validation/lions'))

create_train_val_dirs(root_dir)

def print_subdirectories(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for name in dirs:
            print(os.path.join(root, name))

print_subdirectories(root_dir)

def split_data(SOURCE_DIR, TRAIN_DIR, VALIDATION_DIR, SPLIT_SIZE):
    all_files = []
    for file in os.listdir(SOURCE_DIR):
        if os.path.getsize(os.path.join(SOURCE_DIR, file)) > 0:
            all_files.append(file)
        else:
            print(f'{file} is zero length, so ignoring')
    
    all_files = random.sample(all_files, len(all_files))
    SPLIT_POINT = int(len(all_files)*SPLIT_SIZE)
    train_files = all_files[:SPLIT_POINT]
    validation_files = all_files[SPLIT_POINT:]

    for file in train_files:
        src = os.path.join(SOURCE_DIR, file)
        dest = os.path.join(TRAIN_DIR, file)
        copyfile(src, dest)
    
    for file in validation_files:
        src = os.path.join(SOURCE_DIR, file)
        dest = os.path.join(VALIDATION_DIR, file)
        copyfile(src, dest)

CHEETAH_SOURCE_DIR = './images/Cheetahs'
LION_SOURCE_DIR = './images/Lions'
CHEETAH_TRAIN_DIR = './data/training/cheetahs'
LION_TRAIN_DIR = './data/training/lions'
CHEETAH_VALIDATION_DIR = './data/validation/cheetahs'
LION_VALIDATION_DIR = './data/validation/lions'
split_size = 0.9

split_data(CHEETAH_SOURCE_DIR, CHEETAH_TRAIN_DIR, CHEETAH_VALIDATION_DIR, split_size)
split_data(LION_SOURCE_DIR, LION_TRAIN_DIR, LION_VALIDATION_DIR, split_size)

print(f'Original cheetah images: {len(os.listdir(CHEETAH_SOURCE_DIR))}')
print(f'Original lion images: {len(os.listdir(LION_SOURCE_DIR))}')
print(f'Training cheetah images: {len(os.listdir(CHEETAH_TRAIN_DIR))}')
print(f'Training lion images: {len(os.listdir(LION_TRAIN_DIR))}')
print(f'Validation cheetah images: {len(os.listdir(CHEETAH_VALIDATION_DIR))}')
print(f'Validation lion images: {len(os.listdir(LION_VALIDATION_DIR))}')

def train_val_gens(TRAINING_DIR, VALIDATION_DIR):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow_from_directory(
        TRAINING_DIR,
        batch_size=20,
        class_mode='binary',
        target_size=(150, 150)
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        batch_size=20,
        class_mode='binary',
        target_size=(150, 150)
    )

    return train_generator, validation_generator

train_generator, validation_generator = train_val_gens('./data/training', './data/validation')

def create_model():
    pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                    include_top=False,
                                    weights='imagenet')
    
    for layer in pre_trained_model.layers:
        layer.trainable = False

    last_layer = pre_trained_model.get_layer('mixed7')
    last_output = last_layer.output

    x = tf.keras.layers.Flatten()(last_output)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.models.Model(pre_trained_model.input, x)

    model.compile(
        optimizer=RMSprop(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

model = create_model()

history = model.fit(train_generator,
                    epochs=15,
                    validation_data=validation_generator,
                    verbose=1
                    )