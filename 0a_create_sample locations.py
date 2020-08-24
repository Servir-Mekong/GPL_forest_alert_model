import ee
from utils import sample_point_generator

ee.Initialize()

if __name__ == "__main__":
    
    # Define an arbitrary geometry
    input_study_area = ee.Geometry.Polygon([[[104.3424, 14.2134],[104.3424, 12.2591],[106.0673, 12.2591],[106.0673, 14.2134]]], None, False);
    input_x_cuts = 3
    input_y_cuts = 4
    input_num_samples = 1000
    input_kernel_size = 256
    input_raster_resolution = 10
    input_projection = ee.Projection('EPSG:32648')
    export_asset_directory = 'users/JohnBKilbride/SERVIR/real_time_monitoring' 
    
    # Instantiate the thing
    grid_generator = sample_point_generator.GenerateSamplingGrid(input_study_area, input_x_cuts, input_y_cuts, 
                                                                 input_num_samples, input_kernel_size, input_raster_resolution, 
                                                                 input_projection, export_asset_directory)
    
    # Run the script
    grid_generator.generate_sample_grid()
    
    print('Program is complete')
    