# Stop Pandas from complaining
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# All other libraries
import os
import zipfile
import glob
import rasterio
import shutil
import geopandas as gpd
from rasterio.merge import merge
from rasterio.windows import Window
from rasterio.windows import get_data_window

def get_list_of_zipped_files (directory):
    '''Get a list of all of the zip files in the directory'''
    zip_files = []
    for item in os.listdir(directory): # loop through items in dir
    
        # check for ".zip" extension
        if item.endswith('.zip'): 
        
            # Append to the image
            zip_files.append(directory+'\\'+item) 
            
    return zip_files

def find_image_tiffs (directory, search_pattern):
    '''Get a list of all of the tifs in the directory'''
    tifs = glob.glob(directory + search_pattern, recursive = True)
    return tifs

def extact_export_name (input_path):
    '''Extract the file name from the input file path'''
    # Split based on the back slashes
    split_dashes = zipped_file.split("\\")
    
    # Split based on the underscore
    split_underscores = split_dashes[-1].split("_")

    # Select the components we need    
    file_name = split_underscores[0] + "_" + split_underscores[1] + '.tif'
    
    return file_name

def mosaic_list_of_images (image_paths, shp_path):
    '''
    Takes a list of image paths and mosaics the files using Rasterio's merge function.
    
    returns:
        mosaic    - the mosaic'd array of input images
        out_trans - the output transformation
        out_meta  - the output metadata
        
    ''' 
    # Open the files within the data
    src_files_to_mosaic = []
    for image_path in image_paths:
        source_file = rasterio.open(image_path)
        src_files_to_mosaic.append(source_file)
       
    # Load in the perimeter's geometry
    shp_bounds = perimeter_to_bounding_box(shp_path, source_file.crs.data)
        
    # Merge the mosaics
    mosaic, transformation = merge(src_files_to_mosaic, bounds = shp_bounds)
    
    # Copy the metadata
    out_meta = source_file.meta.copy()
    
    # Update the metadata
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": transformation,
        "crs": source_file.crs
        })
    
    return mosaic, out_meta

def perimeter_to_bounding_box (input_path, crs_data):
    '''Takes a path to a shapefile, opens it, gets the bounds of the perimeter'''
    # Load the shapefile's perimeter
    perimeter_geo = gpd.read_file(input_path)
    
    # Reproject the perimeter intot he same coordinate system as the composite
    perimeter_bounds = perimeter_geo.to_crs(crs=crs_data).bounds
    
    return tuple(perimeter_bounds.to_numpy()[0])

def create_temporary (path):
    '''Create a temporary directory'''
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    return None

def write_image_dataset (mosaic, output_metadata, output_path):
    '''
    Write the output mosaic
    
    Paramters:
        mosaic          - The output array produced by rasterio merge
        output_metadata - The output metadata
        output_path     - The output file path
    
    '''
    # Write the mosaic raster to disk
    with rasterio.open(output_path, "w", **output_metadata) as dest:
        dest.write(mosaic) 
    return None

def get_dict_of_shp_files(directory):
    '''Get a list of all of the shapefiles files in the directory'''
    zip_files_dict = {}
    for item in os.listdir(directory): # loop through items in dir
    
        # check for ".zip" extension
        if item.endswith('.shp'): 
        
            # Append to the image
            zip_files_dict[item.split('.')[0]] = directory+'\\'+item 
            
    return zip_files_dict

if __name__ == '__main__':
    
    # Specificy the directory directories
    zip_dir = 'E:\\temp'
    temp_dir = 'C:\\Users\\John Kilbride\\Desktop\\temp_ssd_space\\temporary'
    output_dir = 'C:\\Users\\John Kilbride\\Desktop\\temp_ssd_space\\tiffs'
    
    # Define the image size
    image_size = 64

    # Get a list of the zipped files
    zipped_files = get_list_of_zipped_files(zip_dir)
   
    # Run the extraction for all of the images
    print('Beginning processing...')
    for i, zipped_file in enumerate(zipped_files):

        print('Processing: ', i+1, ' of ', len(zipped_files))
    
        # Create the temporary directory
        create_temporary(temp_dir)
        
        # Unzip the current zipped_file to the target directory
        zip_ref = zipfile.ZipFile(zipped_file) 
        zip_ref.extractall(temp_dir) # extract file to dir
        zip_ref.close() # close file
        
        # Aggregate all of the correct images within the temporary directory
        tifs = find_image_tiffs(temp_dir, "/**/*.tif")
        
        # Move the files to the target directory
        for cur_path in tifs:
        
            with rasterio.open(cur_path) as src:
        
                # Define the window
                window = Window(col_off=0, row_off=0, width=image_size, height=image_size)
            
                # Get and update the metadata
                kwargs = src.meta.copy()
                kwargs.update({
                    'height': window.height,
                    'width': window.width,
                    'transform': rasterio.windows.transform(window, src.transform)})
            
                with rasterio.open(output_dir + '\\' + cur_path.split('\\')[-1], 'w', **kwargs) as dst:
                    dst.write(src.read(window=window))
    
    # Delete the temporary directory
    shutil.rmtree(temp_dir)
                    
    print('\nProgram completed.')
