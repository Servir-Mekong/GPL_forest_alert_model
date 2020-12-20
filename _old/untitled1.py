import ee

ee.Initialize()

class GenerateSamplingGrid():
    '''
    Author: John Kilbride - john.b.kilbride@gmail.com
    
    Description: 
    
    The output sample size will approximately be x_cuts * y_cuts * input_num_samples * 8. As num samples increases
    the accuracy of the approximation will improve. This is due to rounding that occurs when laying out a grid 
    that fits in the partition. 
    
    
    Parameters [name (type): description]:
    study_area (geometry): An ee.Geometry of the target area. The bounds will be uses. 
    x_cut (int): The number of splits perpendicular to the x-axis (vertical splits).
    y_cuts (int): The number of splits perpendicular to the y-axis (vertical splits). 
    num_samples (int): The target number of outputs, not because of the tiling scheme I used, you may not get exactly that many
    sample points. Fixing this will take forever so let's just all agree to use what it outputs!
    feature_size (int): the size of the FCNN input width and height (e.g., 64, 265)
    raster_resolution (int): The spatial resolution (meters) of the satellite inputs.
    projection (ee.Projection): A projected coordinate system to use (use "feet" at your own risk). 
    
    Outputs: 
    partitions (ee.FeatureCollection):
    sample_locations (ee.FeatureCollection): Total sample s
    
    '''
    
    def __init__(self, study_area, x_cuts, y_cuts, target_sample_size, kernel_size, raster_resolution, projection, asset_directory):
        
        self.study_area = study_area
        self.x_cuts = x_cuts
        self.y_cuts = y_cuts
        self.target_sample_size = target_sample_size
        self.kernel_size = kernel_size
        self.raster_resolution = raster_resolution
        self.projection = projection
        self.asset_directory = asset_directory
        
        return None

    # Contains the main logic of the script
    def generate_sample_grid (self):
        
        print('Generating samples...')
        
        # Project the geometry into the target projection
        bounding_box = self.study_area.bounds(ee.ErrorMargin(1, 'projected'), self.projection) \
            .transform(self.projection) \
            .bounds(ee.ErrorMargin(1, 'projected'), self.projection)
        
        # Get the bounding box partitions
        partitions = self.__partition_bounding_box(bounding_box)
        
        # Export all of the various ee.FeatureCollesctions to the input asset directory
        self.__export_compute_feature_collections(partitions, sample_points, sample_tiles)
        print('Output number of Samples:', sample_points.size().getInfo())
        
        return None

    # Divide the geometry into pieces
    def __partition_bounding_box (self, box):
        
        # Compute the erosion distance to shrink the partitions
        erosion_dist =  self.kernel_size * self.raster_resolution
        
        # Get the necessary coordinates from the bounding box
        coords = ee.Array(box.coordinates())
        
        # Assign the coordinates to each corner
        xy_0_0 = ee.Geometry.Point(coords.slice(1,0,1).reshape([-1]).toList(), self.projection)
        xy_1_1 = ee.Geometry.Point(coords.slice(1,2,3).reshape([-1]).toList(), self.projection)
        
        # Get the distance between The LLH corner and the URH corner
        delta_x_y = ee.Array(xy_1_1.coordinates()).subtract(ee.Array(xy_0_0.coordinates()))
        delta_x = delta_x_y.get([0]).divide(self.x_cuts)
        delta_y = delta_x_y.get([1]).divide(self.y_cuts)
        
        # Create a feature collection of all of the verticies 
        # Loop throuhg the y coordinates in the outer loop and
        # loop through the x coordinates in the inner loop
        xy_pairs = []
        for y in range(0, self.y_cuts+1):
            for x in range(0, self.x_cuts+1):
                xy_pairs.append([x,y])
        
        # Convert the xy pairs into points
        # Note the function requires scope considerations...
        def xy_to_points (pair):
            
            # Get the x and y values
            x_coord = ee.Number(ee.List(pair).get(0))
            y_coord = ee.Number(ee.List(pair).get(1))
            
            # Translate the origin to obtain the new point
            x_translate = delta_x.multiply(x_coord).multiply(-1)
            y_translate = delta_y.multiply(y_coord).multiply(-1)
            translate_prj = self.projection.translate(x_translate, y_translate)
            point_geo = ee.Geometry.Point(xy_0_0.transform(translate_prj).coordinates(), self.projection)
            
            # Create the new feature that needs to be exported
            feature = ee.Feature(point_geo, {'grid_x':x_coord, 'grid_y':y_coord})
            
            return feature
        
        # Turn the client-side list of xy-pairs into a FeatureCollection of points
        grid_vertices = ee.FeatureCollection(ee.List(xy_pairs).map(xy_to_points))
        
        # Create a list of the starting vertices from which to generate the partitions
        partition_pairs = []
        partition_id = 0
        for y in range(0, self.y_cuts):
            for x in range(0, self.x_cuts):
                partition_pairs.append([x, y, partition_id])
                partition_id += 1  
        
        # Create the partions from the partitions pairs lists
        def partitions_pairs_to_partitions (pair):
        
            # Get the parameters from the input array
            x_coord = ee.Number(ee.List(pair).get(0)).toInt16()
            y_coord = ee.Number(ee.List(pair).get(1)).toInt16()
            id_str = ee.Number(ee.List(pair).get(2)).toInt16()
            
            # Retrieve the 4 verticies needed
            llh = ee.Feature(grid_vertices.filterMetadata('grid_x', 'equals', x_coord) \
                .filterMetadata('grid_y', 'equals', y_coord).first()).geometry()
            lrh = ee.Feature(grid_vertices.filterMetadata('grid_x', 'equals', x_coord.add(1)) \
                .filterMetadata('grid_y', 'equals', y_coord).first()).geometry()
            urh = ee.Feature(grid_vertices.filterMetadata('grid_x', 'equals', x_coord.add(1)) \
                .filterMetadata('grid_y', 'equals', y_coord.add(1)).first()).geometry()
            ulh = ee.Feature(grid_vertices.filterMetadata('grid_x', 'equals', x_coord) \
                .filterMetadata('grid_y', 'equals', y_coord.add(1)).first()).geometry()
            
            # Convert into a geometry
            partition_geo = ee.Geometry.Polygon([[llh.coordinates(), lrh.coordinates(), urh.coordinates(), ulh.coordinates()]], self.projection, False)
            
            # Append the feature to the output
            partition = ee.Feature(partition_geo, {'partition_id':id_str, 'partition_x':x_coord, 'partition_y':y_coord})
            
            return partition
        
        partitions = ee.FeatureCollection(ee.List(partition_pairs).map(partitions_pairs_to_partitions))
        
        return partitions
    
    def __compute_grid_constants (self, sample_partition, num_samples, prj):
        '''
        Compute the constants needed in the process of the grid generation.
        '''
        # For now just get one partition
        partition = ee.Feature(sample_partition)
        
        # Get the necessary coordinates from the partition
        coords = ee.Array(partition.geometry().coordinates())
        
        # Assign the coordinates to each corner
        xy_0_0 = ee.Geometry.Point(coords.slice(1,0,1).reshape([-1]).toList(), prj)
        xy_1_1 = ee.Geometry.Point(coords.slice(1,2,3).reshape([-1]).toList(), prj)
        
        # Get the distance between The LLH corner and the URH corner
        delta_x_y = ee.Array(xy_1_1.coordinates()).subtract(ee.Array(xy_0_0.coordinates()))
        width = delta_x_y.get([0])
        height = delta_x_y.get([1])
        
        # Compute number of columns (nx) and the number of rows (ny)
        nx = self.__calculate_sample_points_x(width, height, num_samples)
        ny = ee.Number(num_samples).divide(nx)
        
        # Round the two values
        nx = nx.round().getInfo()
        ny = ny.round().getInfo()
        
        return [width, height, nx, ny]
    # Run the export functions
    def __export_compute_feature_collections(self, partitions):
    
        # Create all of the assetIds for the export functions
        partition_asset_path = self.asset_directory+'/partitions'
        
        # Export the partitions
        task_1 = ee.batch.Export.table.toAsset(
            collection = partitions,
            description = 'Export-Table-Partitions', 
            assetId = partition_asset_path
            )

        #Start the tasks
        task_1.start()
        
        return None

if __name__ == "__main__":
    
    # Define an arbitrary geometry
    input_study_area = ee.Geometry.Polygon([[[104.0311, 14.3134],[104.0311, 12.5128],[106.0416, 12.5128],[106.0416, 14.3134]]], None, False)
    input_x_cuts = 5
    input_y_cuts = 5
    input_num_samples = 4
    input_kernel_size = 256
    input_raster_resolution = 10
    input_projection = ee.Projection('EPSG:32648')
    export_asset_directory = 'users/JohnBKilbride/SERVIR/real_time_monitoring' 
    
    # Instantiate the thing
    grid_generator = GenerateSamplingGrid(input_study_area, input_x_cuts, input_y_cuts, 
                                          input_num_samples, input_kernel_size, input_raster_resolution, 
                                          input_projection, export_asset_directory)

    # Run the script
    grid_generator.generate_sample_grid()
    
    print('Program is complete')
    