
import os
import time
import numpy as np
from glob import glob
import json
import random
import cv2
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
policy = tf.keras.mixed_precision.Policy("mixed_float16")
tf.keras.mixed_precision.set_global_policy(policy)
strategy = tf.distribute.MultiWorkerMirroredStrategy()
from tensorflow import keras
from tensorflow.keras import layers
from keras_cv_attention_models import efficientnet

from utils import overcomplicated

cwd = os.getcwd()
image_frames = 60
image_size = 384
input_shape = (image_frames, image_size, image_size, 3)
label_data_root = cwd + '/dataset/universal_labels'
image_data_root = cwd + '/dataset/features512'
validation_image_data_root = cwd + '/dataset/features512_validation'
validation_new_label_data_root = cwd +  '/dataset/universal_labels_validation'
batch_size = 32

# parse arguments
import argparse
parser = argparse.ArgumentParser(description='Train a model.')
parser.add_argument('--image_frames', type=int, default=60, help='number of frames per video')
parser.add_argument('--image_size', type=int, default=384, help='size of image')
parser.add_argument('--label_data_root', type=str, default=cwd + '/dataset/universal_labels', help='path to label data')
parser.add_argument('--image_data_root', type=str, default=cwd + '/dataset/features512', help='path to image data')
parser.add_argument('--validation_image_data_root', type=str, default=cwd + '/dataset/features512_validation', help='path to validation image data')
parser.add_argument('--validation_new_label_data_root', type=str, default=cwd +  '/dataset/universal_labels_validation', help='path to validation label data')
parser.add_argument('--batch_size', type=int, default=32, help='batch size, use 16 per 24gb of vram')

args = parser.parse_args()
image_frames = args.image_frames
image_size = args.image_size
input_shape = (image_frames, image_size, image_size, 3)
label_data_root = args.label_data_root
image_data_root = args.image_data_root
validation_image_data_root = args.validation_image_data_root
validation_new_label_data_root = args.validation_new_label_data_root
batch_size = args.batch_size


class mystoppingCB(tf.keras.callbacks.Callback):     # sometimes I want it to stop training
    def on_epoch_end(self, epoch, logs=None):        # so I made this callback
        if os.path.exists('epochstop.txt'):          # because I run it in a notebook and 
            self.model.stop_training = True          # already queued up the next cell
            os.remove('epochstop.txt')

validation_gen = overcomplicated.datasetloader(validation_new_label_data_root, validation_image_data_root, duration=60, batch_size=2, croppct=0, image_size=image_size)
train_gen = overcomplicated.datasetloader(label_data_root, image_data_root, duration=image_frames, batch_size=batch_size, croppct=0.2, augment=True, image_size=image_size)

with strategy.scope():
    models = glob('models/*.h5')
    if len(models) > 0:
        models.sort(key=os.path.getmtime)
        model_name = models[-1]
        print('loading model: ' + model_name)
        model = keras.models.load_model(model_name)
        model.summary()
    else:
        backbone_file = 'efficientnetv2-s-21k-ft1k.h5'
        if not os.path.exists(backbone_file):
            print('downloading backbone')
            os.system('wget https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/' + backbone_file + ' -O ' + backbone_file)
        backbone = efficientnet.EfficientNetV2S(pretrained=backbone_file,dropout=1e-6, num_classes=0, include_preprocessing = True)
        backbone.summary()
        backbone.trainable = False
        inputs = keras.Input(shape=input_shape)
        backbone_inputs = keras.Input(shape=(image_size, image_size, 3))
        y = backbone(backbone_inputs)
        y = layers.Flatten()(y)
        y = layers.Dense(64, activation="relu")(y)
        y = layers.Dropout(0.1)(y)
        x = layers.TimeDistributed(keras.Model(backbone_inputs, y))(inputs)
        x = layers.Dropout(0.1)(x)
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(image_frames, activation="relu")(x)
        model = keras.Model(inputs, outputs)
        print(model.summary())
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="mean_squared_error",
            metrics=["mean_squared_error", "mean_absolute_error"]
        )
        model.save('models/base_' + str(image_frames) + "_" + str(image_size) + '_' + backbone_file)

    callbacks = [
                    tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(time.time()), update_freq='batch'),
                    tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/{}'.format(time.time()), save_weights_only=False, verbose=0, save_freq='epoch'),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.7, patience=3, verbose=1, mode='min', min_delta=0.0001, cooldown=0, min_lr=0),
                    mystoppingCB()
                    
    ]
    model.optimizer.learning_rate = 1e-3
    try:
        model.fit(train_gen, validation_data=validation_gen, epochs=1000, callbacks=callbacks, verbose=1)
    except KeyboardInterrupt:
        pass
    # save in h5
    model.save('models/' + str(time.time()) + '.h5')
    # save in keras
    model.save('models/' + str(time.time()) + '.keras')




    