import torch
import rasterio
from glob import glob

def load_geotiff_as_array(geotiff_path, channels, kernel_size):
    '''
    Returns an array with the structure: [row number, column number, band, year]
    '''
    #Read the Multi-band raster values into a 3D array
    #Array structure ['band', 'row number', 'column num']
    with rasterio.open(geotiff_path, 'r') as ds:
        image_data = ds.read()
    
    # This corresponds to [channel, row number, column number]
    reshaped = image_data[:-1, 0:kernel_size, 0:kernel_size]
    
    return torch.Tensor(reshaped).reshape(channels, kernel_size*kernel_size).double()

if __name__ == '__main__':
    
    # Defien a path to a dataset that needs to be laoded
    files = glob("C:\\Users\\John Kilbride\\Desktop\\temp_ssd_space\\tiffs\\*tif")
    
    # Define the number of channels in the input features
    num_features = 9
    
    # Define the kernel size
    kernel_size = 64
    
    # Loop through the files
    num_records = len(files)
    mean = torch.Tensor([0,0,0,0,0,0,0,0,0]).double()
    std = torch.Tensor([0,0,0,0,0,0,0,0,0]).double()
        
    print('Running...')
    for i, file in enumerate(files):
        
        if i % 2500 == 0 and i != 0:
            print('...still running...')

        # Load the image as a tensor
        pt_tensor = load_geotiff_as_array(file, num_features, kernel_size)
        
        # Compute the mean and the standard deviations
        mean += pt_tensor.mean(1)
        std += pt_tensor.std(1)
    
    # Scale em
    mean = mean / num_records
    std = std / num_records
    
    
    mean = [round(num, 3) for num in mean.numpy().tolist()]
    std = [round(num, 3) for num in std.numpy().tolist()]
    
    print("\nMean:     ", mean)
    print("Std. Dev. :", std)
    
    print("\nProgram Completed.")
    
    