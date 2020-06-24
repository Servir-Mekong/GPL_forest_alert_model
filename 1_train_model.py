import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import metrics
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import backend as K

from utils import custom_metrics

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

def get_dataset(pattern):
    """Function to read, parse and format to tuple a set of input tfrecord files.
    Get all the files matching the pattern, parse and convert to tuple.
    Args:
        pattern: A file pattern to match in a Cloud Storage bucket.
    Returns:
        A tf.data.Dataset
    """
    glob = tf.io.gfile.glob(pattern)
    dataset = tf.data.TFRecordDataset(glob, compression_type='GZIP')
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=8)
    dataset = dataset.map(to_tuple, num_parallel_calls=8)
    return dataset

def get_training_dataset(input_path):
    """Loads the training dataset exported by GEE
    Returns: 
        A tf.data.Dataset of training data.
    """
    dataset = get_dataset(input_path+'Train*')
    return dataset
    
def get_testing_dataset(input_path):
    """Loads the test dataset exported by GEE
        Returns: 
        A tf.data.Dataset of evaluation data.
    """
    dataset = get_dataset(input_path+'Test*')
    return dataset

def get_dataset_length(dataset):
    '''Returns the length of a TFrecord dataset. Praise be TFRecords...'''
    num_elements = 0
    for element in dataset:
        num_elements += 1
    return num_elements

def get_modeling_data_stats(train, test):
    '''
    Computes the channel means and the std of the train and test sets.
    Additionall, the length of the training and testing sets are computed
    
    Paramters:
        train: a TFRecordDataset for the training set
        test: a TFRecordDataset for the testing set
    
    Returns:
        
    
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


def jaccard_distance(y_true, y_pred, smooth=100):
    """Jaccard distance for semantic segmentation.
    Also known as the intersection-over-union loss.
    This loss is useful when you have unbalanced numbers of pixels within an image
    because it gives all classes equal weight. However, it is not the defacto
    standard for image segmentation.
    For example, assume you are trying to predict if
    each pixel is cat, dog, or background.
    You have 80% background pixels, 10% dog, and 10% cat.
    If the model predicts 100% background
    should it be be 80% right (as with categorical cross entropy)
    or 30% (with this loss)?
    The loss has been modified to have a smooth gradient as it converges on zero.
    This has been shifted so it converges on 0 and is smoothed to avoid exploding
    or disappearing gradient.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    # Arguments
        y_true: The ground truth tensor.
        y_pred: The predicted tensor
        smooth: Smoothing factor. Default is 100.
    # Returns
        The Jaccard distance between the two tensors.
    # References
        - [What is a good evaluation measure for semantic segmentation?](
            http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf)
    
    Source: 
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def conv_block(input_tensor, num_filters):
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    return encoder

def encoder_block(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
    return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters):
    decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    return decoder

def get_model(input_optimizer, input_loss_function, evaluation_metrics):
    inputs = layers.Input(shape=[None, None, len(BANDS)]) # 256
    encoder0_pool, encoder0 = encoder_block(inputs, 32) # 128
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64) # 64
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128) # 32
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256) # 16
    encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512) # 8
    center = conv_block(encoder4_pool, 1024) # center
    decoder4 = decoder_block(center, encoder4, 512) # 16
    decoder3 = decoder_block(decoder4, encoder3, 256) # 32
    decoder2 = decoder_block(decoder3, encoder2, 128) # 64
    decoder1 = decoder_block(decoder2, encoder1, 64) # 128
    decoder0 = decoder_block(decoder1, encoder0, 32) # 256
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)
    
    model = models.Model(inputs=[inputs], outputs=[outputs])
    
    model.compile(
        optimizer = input_optimizer, 
        loss = input_loss_function,
        metrics = evaluation_metrics
        )
    
    return model

if __name__ == "__main__":
    
    # Set the path to the raw data
    raw_data_path = ".\\raw_data\\"
    
    # Define the path to the log directory for tensorboard
    log_dir = '.\\logs'
    
    # Define the directory where the models will be saved
    model_dir = '.\\models'
    
    # Specify inputs (Landsat bands) to the model and the response variable.
    BANDS = ['VH_pre','VV_pre','VH_post','VV_post']
    RESPONSE = 'label'
    FEATURES = BANDS + [RESPONSE]
    
    # Specify the size and shape of patches expected by the model.
    kernel_size = 256
    kernel_shape = [kernel_size, kernel_size]
    COLUMNS = [tf.io.FixedLenFeature(shape=kernel_shape, dtype=tf.float32) for k in FEATURES]
    FEATURES_DICT = dict(zip(FEATURES, COLUMNS))

    # Load the two raw datasets
    raw_training_ds = get_training_dataset(raw_data_path)
    raw_testing_ds = get_testing_dataset(raw_data_path)
    
    # Compute the stats needed for normalization
    NORM_MEAN, NORM_STD, TRAIN_SIZE, TEST_SIZE = get_modeling_data_stats(raw_training_ds, raw_testing_ds)
    
    # Normalize the training adn the testing sets to have zero mean and unit variance
    training_ds = raw_training_ds.map(apply_normalization, num_parallel_calls=8)
    testing_ds = raw_testing_ds.map(apply_normalization, num_parallel_calls=8)

    # Specify model training parameters.
    BATCH_SIZE = 16
    EPOCHS = 20
    BUFFER_SIZE = 2000
    optimizer = 'SGD'
    eval_metrics = ['Accuracy', custom_metrics.dice_coef, custom_metrics.f1_m, 
                    custom_metrics.precision_m, custom_metrics.recall_m]

    # Load the model -- Currently Google's UNET
    model = get_model(optimizer, jaccard_distance, eval_metrics)

    # Fit the model to the data
    model.fit(
        x = training_ds, 
        epochs = EPOCHS, 
        steps_per_epoch = int(TRAIN_SIZE / BATCH_SIZE), 
        validation_data = testing_ds,
        validation_steps = TEST_SIZE,
        callbacks = [tf.keras.callbacks.TensorBoard(log_dir)]
        )
    
    # Save the model
    model.save(model_dir, save_format='tf')