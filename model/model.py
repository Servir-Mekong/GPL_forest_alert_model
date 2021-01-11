import os
import sys
import numpy as np
import tensorflow as tf
import glob

from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import metrics
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import UpSampling2D, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, \
     Add, Dense, BatchNormalization, ReLU, MaxPool2D, GlobalAvgPool2D, Activation

sys.path.append("utils")
import dataio, custom_metrics


def parse_tfrecord(example_proto):
    """The parsing function.
    Read a serialized example into the structure defined by FEATURES_DICT.
    Args:
        example_proto: a serialized Example.
    Returns:
        A dictionary of tensors, keyed by feature name.
    """
    return tf.io.parse_single_example(example_proto, FEATURES_DICT)

def to_tuple(inputs):
    """Function to convert a dictionary of tensors to a tuple of (inputs, outputs).
    Turn the tensors returned by parse_tfrecord into a stack in HWC shape.
    Args:
        inputs: A dictionary of tensors, keyed by feature name.
    Returns:
        A tuple of (inputs, outputs).
    """
    inputsList = [inputs.get(key) for key in FEATURES]
    stacked = tf.stack(inputsList, axis=0)
    # Convert from CHW to HWC
    stacked = tf.transpose(stacked, [1, 2, 0])
    return stacked[:,:,:len(BANDS)], stacked[:,:,len(BANDS):]


def get_modeling_data_stats(train, test):
    '''
    Computes the channel means and the std of the train and test sets.
    Additionall, the length of the training and testing sets are computed
    
    Paramters:
        train: a TFRecordDataset for the training set
        test: a TFRecordDataset for the testing set
    
    Returns:
        mean - a tf.Tensor(4,) containing the channel means
        std - a tf.Tensor(4,) containin the channel standard deviation
        train_len - the number of elements in the training dataset
        test_len - the number of elements in the testing dataset
    
    '''
    # Initialize the variables that we need to keep track of
    mean = tf.constant([0.,0.,0.,0.])
    std = tf.constant([0.,0.,0.,0.])
    nb_samples = 0.0
    train_len = 0.0
    test_len = 0.0
    
    # Loop through the training dataset
    for element in train:
        mean = tf.math.add(mean, tf.math.reduce_mean(element[0], axis=[0,1]))
        std = tf.math.add(std, tf.math.reduce_std(element[0], axis=[0,1]))
        nb_samples += 1
        train_len += 1
    
    # Loop through the testing dataset
    for element in test:
        mean = tf.math.add(mean, tf.math.reduce_mean(element[0], axis=[0,1]))
        std = tf.math.add(std, tf.math.reduce_std(element[0], axis=[0,1]))
        nb_samples += 1
        test_len += 1
        
    # Divide by the number of elements in the two sets
    mean = tf.math.divide(mean, nb_samples)
    std = tf.math.divide(std, nb_samples)

    return mean, std, train_len, test_len

def apply_normalization(features, label):
    '''
    Apply the z-score transformation
    Note this uses two global variables 
    '''
    norm = tf.math.divide(tf.math.subtract(features, NORM_MEAN), NORM_STD)
    
    return norm, label

def dice_loss(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    true_sum = K.sum(K.square(y_true), -1)
    pred_sum = K.sum(K.square(y_pred), -1)
    return 1 - ((2. * intersection + smooth) / (true_sum + pred_sum + smooth))


def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = LeakyReLU(alpha=0.1)(x)
    return x

def residual_block(blockInput, num_filters=16):
    x = LeakyReLU(alpha=0.1)(blockInput)
    x = BatchNormalization()(x)
    blockInput = BatchNormalization()(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    return x


def build_model_mobilenet():
    inputs = Input(shape=(None, None, 8), name="input_image")
    
    encoder = MobileNetV2(input_tensor=inputs, weights=None, include_top=False) #, alpha=0.35)
    skip_connection_names = ["input_image", "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu"]
    encoder_output = encoder.get_layer("block_13_expand_relu").output
    
    #f = [32, 64, 128, 256]
    #f = [4, 8, 16, 32]
    f = [8, 16, 32, 64]
    x = encoder_output
    for i in range(1, len(skip_connection_names)+1, 1):
        x_skip = encoder.get_layer(skip_connection_names[-i]).output
        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_skip])
        
        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
    x = Conv2D(2, (2, 2), padding="same")(x)
    x = Activation("sigmoid")(x)
    
    model = Model(inputs, x)
    return model


def build_model_xception(start_neurons=1):
    
    backbone = Xception(input_shape=(None, None,8), weights=None, include_top=False)
    input = backbone.input
    
    conv4 = backbone.layers[121].output
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.1)(pool4)
    
     # Middle
    convm = Conv2D(start_neurons*32, (3, 3), activation='relu', padding="same")(pool4)
    convm = residual_block(convm, start_neurons*32)
    convm = residual_block(convm, start_neurons*32)
    convm = LeakyReLU(alpha=0.1)(convm)
    
    # 8 -> 16
    deconv4 = Conv2DTranspose(start_neurons*16, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.1)(uconv4)
    
    uconv4 = Conv2D(start_neurons*16, (3, 3), activation='relu', padding="same")(uconv4)
    uconv4 = residual_block(uconv4, start_neurons * 16)
    uconv4 = residual_block(uconv4, start_neurons*16)
    uconv4 = LeakyReLU(alpha=0.1)(uconv4)
    
    # 16 -> 32
    deconv3 = Conv2DTranspose(start_neurons*8, (3, 3), strides=(2, 2), padding="same")(uconv4)
    conv3 = backbone.layers[31].output
    uconv3 = concatenate([deconv3, conv3])    
    uconv3 = Dropout(0.1)(uconv3)
    
    uconv3 = Conv2D(start_neurons*8, (3, 3), activation='relu', padding="same")(uconv3)
    uconv3 = residual_block(uconv3, start_neurons*8)
    uconv3 = residual_block(uconv3, start_neurons*8)
    uconv3 = LeakyReLU(alpha=0.1)(uconv3)

    # 32 -> 64
    deconv2 = Conv2DTranspose(start_neurons*4, (3, 3), strides=(2, 2), padding="same")(uconv3)
    conv2 = backbone.layers[21].output
    conv2 = ZeroPadding2D(((1,0),(1,0)))(conv2)
    uconv2 = concatenate([deconv2, conv2])
        
    uconv2 = Dropout(0.1)(uconv2)
    uconv2 = Conv2D(start_neurons*4, (3, 3), activation='relu', padding="same")(uconv2)
    uconv2 = residual_block(uconv2, start_neurons*4)
    uconv2 = residual_block(uconv2, start_neurons*4)
    uconv2 = LeakyReLU(alpha=0.1)(uconv2)
    
    # 64 -> 128
    deconv1 = Conv2DTranspose(start_neurons*2, (3, 3), strides=(2, 2), padding="same")(uconv2)
    conv1 = backbone.layers[11].output
    conv1 = ZeroPadding2D(((3,0),(3,0)))(conv1)
    uconv1 = concatenate([deconv1, conv1])
    
    uconv1 = Dropout(0.1)(uconv1)
    uconv1 = Conv2D(start_neurons*2, (3, 3), activation='relu', padding="same")(uconv1)
    uconv1 = residual_block(uconv1, start_neurons*2)
    uconv1 = residual_block(uconv1, start_neurons*2)
    uconv1 = LeakyReLU(alpha=0.1)(uconv1)
    
    # 128 -> 256
    uconv0 = Conv2DTranspose(start_neurons*1, (3, 3), strides=(2, 2), padding="same")(uconv1)   
    uconv0 = Dropout(0.1)(uconv0)
    uconv0 = Conv2D(start_neurons*1, (3, 3), activation='relu', padding="same")(uconv0)
    uconv0 = residual_block(uconv0, start_neurons*1)
    uconv0 = residual_block(uconv0, start_neurons*1)
    uconv0 = LeakyReLU(alpha=0.1)(uconv0)
    
    uconv0 = Dropout(0.1)(uconv0)
    output_layer_noActi = Conv2D(2, (2,2), padding="same", activation="sigmoid")(uconv0)
    model = Model(inputs=input, outputs=output_layer_noActi)

    return model

def get_model(input_optimizer, input_loss_function, evaluation_metrics):

    model = build_model_xception()
    model.summary()
    model.compile(
        optimizer = input_optimizer, 
        loss = input_loss_function,
        metrics = evaluation_metrics
        )
   
   
    return model

if __name__ == "__main__":
   
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    print("strategy \n")

    # get distributed strategy and apply distribute i/o and model build
    strategy = tf.distribute.MirroredStrategy()

    print('Number of devices: {}'.format(strategy.scope()))

    # Set the path to the raw data
    raw_data_path = r"/datadrive/data/alert/gladAlertv64v4"
    
    # Define the path to the log directory for tensorboard
    log_dir = r'/home/ubuntu/alerts/log'
    
    # Define the directory where the models will be saved
    model_dir = r'/home/ubuntu/alerts/models'
    
    # Specify inputs (Landsat bands) to the model and the response variable.
    FEATURES = ['VH_after0', 'VH_before0', 'VH_before1', 'VH_before2', 'VV_after0', 'VV_before0', 'VV_before1', 'VV_before2']

    LABELS = ["glad_alert","non_alert"]
    
    
    # Specify model training parameters.
    TRAIN_SIZE = 25000
    BATCH_SIZE = 16
    EPOCHS = 18
    BUFFER_SIZE = 8192
    optimizer = 'ADAM'
        
    eval_metrics = [metrics.categorical_accuracy, custom_metrics.f1_m,
                    custom_metrics.precision_m, custom_metrics.recall_m]
    
    # Specify the size and shape of patches expected by the model.
    kernel_size = 64
    kernel_shape = [kernel_size, kernel_size]
    COLUMNS = [tf.io.FixedLenFeature(shape=kernel_shape, dtype=tf.float32) for k in FEATURES]
    FEATURES_DICT = dict(zip(FEATURES, COLUMNS))
    
    KERNEL_SIZE = 64
    PATCH_SHAPE = (KERNEL_SIZE, KERNEL_SIZE)

    # get training, testing, and eval TFRecordDataset
    # training is batched, shuffled, transformed, and repeated
    training = dataio.get_dataset(training_files, FEATURES, LABELS, PATCH_SHAPE, BATCH_SIZE,
                              buffer_size=BUFFER_SIZE, training=True).repeat()
    # for element in training:
    #     print(element)
    #     break
    #
    # iterator = iter(training)
    # print(f'get_next: {iterator.get_next()}')

    # testing is batched by 1 and repeated
    testing = dataio.get_dataset(testing_files, FEATURES, LABELS, PATCH_SHAPE, 1).repeat()
    # eval is batched by 1
    validation = dataio.get_dataset(validation_files, FEATURES, LABELS, PATCH_SHAPE, 1)

    # Load the model -- Currently Google's UNET
    with strategy.scope():
        model = build_model_xception(optimizer, dice_loss, eval_metrics)
        
    
    # Fit the model to the data
    model.fit(
        x = training_ds, 
        epochs = EPOCHS, 
        steps_per_epoch =int(TRAIN_SIZE / BATCH_SIZE), 
        validation_data = testing_ds,
        validation_steps = 200,
        callbacks = [tf.keras.callbacks.TensorBoard(log_dir)],

        )
    
    # Save the model
    model.save(model_dir, save_format='tf')

    # check how the model trained
    my_model.evaluate(validation)
