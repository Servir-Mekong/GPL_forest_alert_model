# Stop Pandas from complaining
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# All other libraries
import os
import zipfile
import glob
import rasterio
import shutil
import numpy as np
import geopandas as gpd
from rasterio.windows import Window
import matplotlib.pyplot as plt
from rasterio.windows import Window
from rasterio.windows import get_data_window

def find_image_tiffs (directory, search_pattern):
    '''Get a list of all of the tifs in the directory'''
    tifs = glob.glob(directory + search_pattern, recursive = True)
    return tifs

if __name__ == "__main__":
    
    # Define the tiff directory
    tiff_dir = 'C:\\Users\\John Kilbride\\Desktop\\temp_ssd_space\\tiffs'
    
    # Define the number of bands/channels in the images (remember to include the label)
    num_channels = 10
    
    # Define the size of each image chip (NxN)
    image_size = 64
    
    # Get a list of all of the GEOTIFFS
    tiff_paths = find_image_tiffs(tiff_dir, '\\*.tif')
    
    # Loop through the GeoTiffs
    print('Trimming rasters...')
    for cur_path in tiff_paths:
    
        with rasterio.open(cur_path) as src:
    
            # Define the window
            window = Window(col_off=0, row_off=0, width=image_size, height=image_size)
        
            # Get the metadata
            kwargs = src.meta.copy()
            kwargs.update({
                'height': window.height,
                'width': window.width,
                'transform': rasterio.windows.transform(window, src.transform)})
        
            with rasterio.open(cur_path, 'w', **kwargs) as dst:
                dst.write(src.read(window=window))
                
    print('\nProgram completed.')