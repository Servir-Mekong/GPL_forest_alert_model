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


# import tensorflow_transform as tft

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
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=5)
    dataset = dataset.map(to_tuple, num_parallel_calls=5)
    return dataset

def get_training_dataset(input_path):
    """Get the preprocessed training dataset
    Returns: 
    A tf.data.Dataset of training data.
    """
    dataset = get_dataset(input_path)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    return dataset
    
def get_eval_dataset(input_path):
    """Get the preprocessed evaluation dataset
        Returns: 
        A tf.data.Dataset of evaluation data.
    """
    dataset = get_dataset(input_path)
    dataset = dataset.batch(1).repeat()
    return dataset

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.dot(y_true, K.transpose(y_pred))
    union = K.dot(y_true,K.transpose(y_true))+K.dot(y_pred,K.transpose(y_pred))
    return (2. * intersection + smooth) / (union + smooth)

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

def normalize_dataset(data, mean_data=None, std_data=None):
    if not mean_data:
        mean_data = np.mean(data)
    if not std_data:
        std_data = np.std(data)
    norm_data = (data-mean_data)/std_data
    return norm_data, mean_data, std_data


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
    
    # # Set the working directory to the file's directory
    # os.chdir(os.path.dirname(sys.argv[0]))
    
    # Define the path to the training and evaluation datasets
    training_path = ".\\training_data\\*"
    evaluation_path = ".\\testing_data\\*"
    
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

    # Sizes of the training and evaluation datasets.
    TRAIN_SIZE = 10045
    EVAL_SIZE = 8785
    
    # Specify model training parameters.
    BATCH_SIZE = 16
    EPOCHS = 20
    BUFFER_SIZE = 2000
    optimizer = 'SGD'
    eval_metrics = ['Accuracy', custom_metrics.dice_coef, custom_metrics.f1_m, 
                    custom_metrics.precision_m, custom_metrics.recall_m]
    
    # Get the training and evaluation datasets
    training = get_training_dataset(training_path)
    evaluation = get_eval_dataset(evaluation_path)

    # Load the model -- Currently Google's UNET
    model = get_model(optimizer, jaccard_distance, eval_metrics)

    # Fit the model to the data
    model.fit(
        x=training, 
        epochs=EPOCHS, 
        steps_per_epoch=int(TRAIN_SIZE / BATCH_SIZE), 
        validation_data=evaluation,
        validation_steps=EVAL_SIZE,
        callbacks=[tf.keras.callbacks.TensorBoard(log_dir)]
        )
    
    # Save the model
    model.save(model_dir, save_format='tf')