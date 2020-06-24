import os
import sys
import numpy as np
import tensorflow as tf

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
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=5)
    dataset = dataset.map(to_tuple, num_parallel_calls=5)
    return dataset

def get_training_dataset(input_path):
    """Loads the training dataset exported by GEE
    Returns: 
        A tf.data.Dataset of training data.
    """
    dataset = get_dataset(raw_data_path+'Train*')
    return dataset
    
def get_testing_dataset(input_path):
    """Loads the test dataset exported by GEE
        Returns: 
        A tf.data.Dataset of evaluation data.
    """
    dataset = get_dataset(raw_data_path+'Test*')
    return dataset

def get_dataset_length(dataset):
    '''Returns the length of a TFrecord dataset. Praise be TFRecords...'''
    num_elements = 0
    for element in dataset:
        num_elements += 1
    return num_elements

def get_normalization_stats(train, test):
    '''Computes the channel means and the std of the train and test sets'''
    # Initialize the variables that we need to keep track of
    mean = tf.constant([0.,0.,0.,0.])
    std = tf.constant([0.,0.,0.,0.])
    nb_samples = 0.0
    
    # Loop through the training dataset
    for element in train:
        mean = tf.math.add(mean, tf.math.reduce_mean(element[0], axis=[0,1]))
        std = tf.math.add(std, tf.math.reduce_std(element[0], axis=[0,1]))
        nb_samples += 1
    
    # Loop through the testing dataset
    for element in test:
        mean = tf.math.add(mean, tf.math.reduce_mean(element[0], axis=[0,1]))
        std = tf.math.add(std, tf.math.reduce_std(element[0], axis=[0,1]))
        nb_samples += 1
        
    # Divide by the number of elements in the two sets
    mean = tf.math.divide(mean, nb_samples)
    std = tf.math.divide(std, nb_samples)

    return mean, std

def apply_normalization(features, label):
    '''
    Apply the z-score transformation
    Note this uses two global variables 
    '''
    norm = tf.math.divide(tf.math.subtract(features, NORM_MEAN), NORM_STD)
    
    return norm, label

if __name__ == "__main__":
    
    # # Set the working directory to the file's directory
    # os.chdir(os.path.dirname(sys.argv[0]))
    
    # Set the path to the raw data
    raw_data_path = ".\\raw_data\\"
    
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
    NORM_MEAN, NORM_STD = get_normalization_stats(raw_training_ds, raw_testing_ds)
    
    # Apply the normalization to each dataset
    training_ds = raw_training_ds.map(apply_normalization, num_parallel_calls=8)
    testing_ds = raw_testing_ds.map(apply_normalization, num_parallel_calls=8)

    