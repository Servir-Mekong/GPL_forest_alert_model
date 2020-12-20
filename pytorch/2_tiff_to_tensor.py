import os
import rasterio
import torch
from glob import glob
from multiprocessing import Pool
import torchvision.transforms as transforms

# Define several global variables
NUM_PROCESSES = 14

# Define the size of the kernel used to extract the values
KERNEL_SIZE = 64

def create_output_directories(path):
    '''Checks if the output directory already exists, it not, the directory is created.'''
    # Check if the primary exists
    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)
 
    # Define the names of the sub-directories tha tneed to exist
    train_path = path + '\\training_data'
    val_path = path + '\\val_data'
    test_path = path + '\\test_data'
    
    # Create the sub-directories (if necessary)
    for path in [train_path, val_path, test_path]:
        if os.path.isdir(path):
            pass
        else:
            os.mkdir(path)        
    
    return {'train': train_path, 'val': val_path, 'test':test_path}

def get_the_tiff_paths(data_path):
    '''
    Returns a list of all of the tiffs that need to be processed.
    The GeoTIFFs are randomly shuffled before being returned
    '''
    # Get a list of all o the paths
    paths = glob(data_path + '\\*.tif')
    
    # Throw an error if no TIFFs ar found
    if len(paths) == 0:
        raise ValueError('No TIFF files found in the data directory (looking for the extension ".tif".')
    
    return paths

def train_val_test_split(tiff_paths):
    '''
    Takes a list of GeoTIFFS and splits the list into a training set, a val
    set, and a testing set.
    '''
    # Seperate the different classes into 3 groups
    train_images = []
    val_images = []
    test_images = []
    
    # Loop through each tiff path
    for tiff_path in tiff_paths:
        
        # Extract the partition ID from the TIFF paths
        dataset_id = tiff_path.split('\\')[-1].split('_')[1]
        
        # Assign the each tiff to the path.
        if dataset_id == 'train':
            train_images.append(tiff_path)
        elif dataset_id == 'validation':
            val_images.append(tiff_path)
        elif dataset_id == 'test':
            test_images.append(tiff_path)
        else:
            raise ValueError('Dataset id is not in ["train", "validation", "test"]')

    # Create the output dictionary
    output = {'train': train_images, 'val': val_images, 'test': test_images}

    return output

def load_geotiff_as_array(geotiff_path):
    '''
    Returns an array with the structure: [row number, column number, band, year]
    '''
    #Read the Multi-band raster values into a 3D array
    #Array structure ['band', 'row number', 'column num']
    with rasterio.open(geotiff_path, 'r') as ds:
        image_data = ds.read()
    
    # This corresponds to [channel, row number, column number]
    reshaped = image_data[:, 0:KERNEL_SIZE, 0:KERNEL_SIZE]
    
    return torch.Tensor(reshaped)

def process_geotiffs(zipped_input_list):
    '''
    Loop through each GeoTIFF path in the list of GeoTIFFs and produces
    data ready to be used with FutureGAN.
    '''
    
    # Iterate through each of the paths in the list of geotiffs
    for input_pair in zipped_input_list:
        
        # Unpack the pair
        geotiff_path = input_pair[0]
        output_dir = input_pair[1]
        
        # Load the GeoTIFF as an a NumPy array
        image_array = load_geotiff_as_array(geotiff_path)
        
        # Check the shape of the image array
        num_rows = image_array.shape[1]
        num_cols = image_array.shape[2]
        
        # IF the image has the incorrect number of dimensions, print image info
        # ELSE export the image data
        if num_rows != KERNEL_SIZE or num_cols != KERNEL_SIZE:
            raise ValueError('\nIncorrect dimensions after reshaping found in image: {}'.format(geotiff_path))
    
        # Define the name for the output file
        output_id_base = geotiff_path.split('\\')[-1][:-4]
    
        # Clamp the values of the tensor
        means = torch.tensor([0.78, 0.651, 0.779, 0.645, -0.001, -0.007, 77.334, 3.185, 162.479, 0])
        stds = torch.tensor([0.05, 0.061, 0.05, 0.063, 0.058, 0.069, 4.622, 1.889, 103.259, 1])
        norm_transform = transforms.Normalize(mean=means, std=stds, inplace=True)
        output_tensor = torch.tensor(norm_transform(image_array), dtype=torch.float32)

        # Construct a name for the output
        output_name = output_id_base + '.pt'
        
        # Construct the output path for the tensor
        write_path = output_dir + '\\' + output_name
        
        # Write out the PyTorch Tensor
        torch.save(output_tensor, write_path)

    return None

if __name__ == '__main__':
    
    # Define the data directory
    data_path = 'C:\\Users\\John Kilbride\\Desktop\\temp_ssd_space\\tiffs'
    
    # Define the number of channels in the feature (sans the label)
    number_channels = 9
    
    # Define the data directory
    output_path = 'C:\\Users\\John Kilbride\\Desktop\\temp_ssd_space\\tensors'

    # Create the output_directories
    dir_dict = create_output_directories(output_path)
    
    # Get a list of all of the files in the image data 
    tiff_paths = get_the_tiff_paths(data_path)
    
    # Split the data into a 80/20 train-val-test sets
    image_path_dict = train_val_test_split(tiff_paths)
        
    # Define the number of processes to start
    pool = Pool(processes=NUM_PROCESSES)   

    # Loop through the train, test, val sets
    print('Beginning processing...')
    for dataset_type in ['train', 'val', 'test']:
        
        # Get the images to process
        image_paths_list = image_path_dict[dataset_type]
        
        # Get the output path
        output_dir = dir_dict[dataset_type]
        
        # Split the tiff paths into sub lists
        image_paths_list_chunks = [image_paths_list[i::NUM_PROCESSES] for i in range(NUM_PROCESSES)]
        
        # Zip stuff up for pooled processing
        zipped = []
        for chunk in image_paths_list_chunks:
            dir_list = [output_dir] * len(chunk)
            zipped.append(list(zip(chunk, dir_list)))
        
        pool.map(process_geotiffs, zipped)
        
    # Close the multiprocessing threads
    pool.close()
    pool.join()

    print('\nScript is complete.')



