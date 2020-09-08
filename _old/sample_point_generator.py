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
        
        # Compute sample size seed
        num_samples = self.target_sample_size / (self.x_cuts * self.y_cuts * 8)
        
        # Project the geometry into the target projection
        bounding_box = self.study_area.bounds(ee.ErrorMargin(1, 'projected'), self.projection) \
            .transform(self.projection) \
            .bounds(ee.ErrorMargin(1, 'projected'), self.projection)
        
        # Get the bounding box partitions
        partitions = self.__partition_bounding_box(bounding_box)
        
        # Get sample centroids
        sample_points = self.__get_sample_points(partitions, num_samples, self.projection)
        
        # Compute the foot-prints for visualization purposes
        sample_tiles = self.__get_sample_tiles(sample_points)
        
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
            partition_geo = ee.Geometry.Polygon([[llh.coordinates(), lrh.coordinates(), urh.coordinates(), ulh.coordinates()]], self.projection, False) \
                .buffer(-1 * erosion_dist/2, ee.ErrorMargin(1, 'projected'), self.projection)
            
            # Append the feature to the output
            partition = ee.Feature(partition_geo, {'partition_id':id_str, 'partition_x':x_coord, 'partition_y':y_coord})
            
            return partition
        
        partitions = ee.FeatureCollection(ee.List(partition_pairs).map(partitions_pairs_to_partitions))
        
        return partitions

    def __calculate_sample_points_x (self, width, height, num_samples):
        '''
        Compute the number of rows needed to nicely tesselate the area
        
        The generation of the grid utilizes ideas presented in this post:
        https://math.stackexchange.com/questions/1039482/how-to-evenly-space-a-number-of-points-in-a-rectangle
        Reviewing the most upvoted answer may be useful for context.
        '''
        # Get the necessary terms
        w = ee.Number(width)
        h = ee.Number(height)
        n = ee.Number(num_samples).toInt16()
        
        # Compute the number of x points (n_x) using the formula given in the source
        nx_term1 = w.divide(h).multiply(n)
        nx_term2 = w.subtract(h).pow(2).divide(h.pow(2).multiply(4))
        nx_term3 = w.subtract(h).divide(h.multiply(2))
        nx = nx_term1.add(nx_term2).sqrt().subtract(nx_term3)
        
        return nx     
    
    def __compute_partition_sample_grid (self, origin, width, height, num_samples, nx, ny, prj, partition_id, partition_x, partition_y):
        '''
        # For an arbitrary rectance - generate a point grid of 1/2 the target size
         and then translate to obtain an overlapping pattern.
        
        The generation of the grid utilizes ideas presented in this post:
        https://math.stackexchange.com/questions/1039482/how-to-evenly-space-a-number-of-points-in-a-rectangle
        Reviewing the most upvoted answer may be useful for context.
        '''
        
        # PROBLEM - Sample size will be incorrect because of rounding on nx and ny terms
        # This can maybe be resolved later...345345345345
        
        # Get the offsets that need to be applied to obtain the grid
        delta_x = width.divide(nx)
        delta_y = height.divide(ny)
        
        # Get the information needed to get the 
        grid_points = []
        grid_id_count = 0
        for y in range(0, ny+1):
            for x in range(0, nx+1):
                grid_points.append([x, y, grid_id_count])
                grid_id_count += 1
        
        # Convert the grid_points parameters into features
        def convert_grid_info_to_points (params):
        
            # Unpack the parameters
            x_coord = ee.Number(ee.List(params).get(0))
            y_coord = ee.Number(ee.List(params).get(1))
            id_num = ee.Number(ee.List(params).get(2))
            
            # Define the new point geometry
            x_translate = delta_x.multiply(x_coord).multiply(-1)
            y_translate = delta_y.multiply(y_coord).multiply(-1)
            translate_prj = prj.translate(x_translate, y_translate)
            point_geo = ee.Geometry.Point(origin.transform(translate_prj).coordinates(), prj)
            
            # Generate the sample point
            sample_point_properties = {
                'grid_point_id': grid_id_count, 
                'grid_x': x, 
                'grid_y':y, 
                'partition_id': id_num,
                'parition_x': x_coord,
                'parition_y': y_coord
                }
            
            sample_point = ee.Feature(point_geo, sample_point_properties)
            
            return sample_point
    
        sample_grid = ee.FeatureCollection(ee.List(grid_points).map(convert_grid_info_to_points))
        
        # Generate a second sample grid that is translated
        grid_points = []
        for y in range(0, ny):
            for x in range(0, nx):
                for i in range(1, 4):
                    grid_points.append([x+(0.25*i), y+(0.25*i), grid_id_count])
                    grid_id_count += 1
                    
        # Fill in the other values of the grid. Document code later... 
        values = [0.25,0.75]
        for y in range(ny, 0, -1):
            for x in range(0, nx):
    
                # Interior points
                for i in range(0, len(values)):
                    grid_points.append([x+values[i], y-values[i], grid_id_count])
                    grid_id_count += 1
            
                grid_points.append([x+0.5, y, grid_id_count])
                grid_id_count += 1
                grid_points.append([x, y-0.5, grid_id_count])
                grid_id_count += 1
            
                # Get the points on the top of the partition
                if x == (nx-1):
                    grid_points.append([x+1, y-0.5, grid_id_count])
                    grid_id_count += 1
                
                # Get the points on the bottom of the partitions
                if y == 1:
                    grid_points.append([x+0.5, y-1, grid_id_count])
                    grid_id_count += 1
            
        sample_grid_translated = ee.FeatureCollection(ee.List(grid_points).map(convert_grid_info_to_points))
        
        # Combine the two feature collections
        output_points = ee.FeatureCollection(sample_grid.merge(sample_grid_translated))
        
        return output_points
    
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
    
    def __get_sample_points (self, partitions, num_samples, prj):
        '''
        Get the centroids for all the image chips that need to be sampled
        '''
        # Precompute several values that we will re-use
        compute_results = self.__compute_grid_constants(ee.Feature(partitions.first()), num_samples, prj)
        width = ee.Number(compute_results[0])
        height = ee.Number(compute_results[1])
        nx = compute_results[2]                  # Not an ee.Number
        ny = compute_results[3]                  # Not an ee.Number
        
        # Loop through values of the 
        num_partitions = partitions.size().toInt16().getInfo()
        partition_ids = ee.List.sequence(0, num_partitions - 1)
        def compute_sample_grids (partition_index):
        
            # Get the current partition
            current_partition = ee.Feature(partitions.filterMetadata('partition_id', 'equals', ee.Number(partition_index)).first())
            
            # Get information from the partition
            partition_id = ee.Number(current_partition.get('partition_id'))
            partition_x = ee.Number(current_partition.get('partition_x'))
            partition_y = ee.Number(current_partition.get('partition_y'))
            
            # Get the necessary coordinates from the partition
            coords = ee.Array(current_partition.geometry().coordinates())
            
            # Assign the coordinates to each corner
            partition_origin = ee.Geometry.Point(coords.slice(1,0,1).reshape([-1]).toList(), prj)
            
            # Within the partition we need to 
            partition_points = self.__compute_partition_sample_grid(partition_origin, width, height, num_samples, nx, ny, 
                                                                    prj, partition_id, partition_x, partition_y)
            
            return partition_points.toList(1e6)
            
        return ee.FeatureCollection(partition_ids.map(compute_sample_grids).flatten())

    # Convert the sample points into tiles that approimate the FCNN inputs
    def __get_sample_tiles (self, sample_points):
        
        # Compute the erosioon distance
        erosion_distance = self.kernel_size * self.raster_resolution
        
        def inner_map (feat):
            return ee.Feature(feat).buffer(erosion_distance/2, ee.ErrorMargin(1,'projected'), self.projection) \
                .bounds(ee.ErrorMargin(1, 'projected'), self.projection)
        return sample_points.map(inner_map)

    # Run the export functions
    def __export_compute_feature_collections(self, partitions, sample_points, sample_tiles):
    
        # Create all of the assetIds for the export functions
        partition_asset_path = self.asset_directory+'/partitions'
        sample_points_asset_path = self.asset_directory+'/partition_points'
        sample_tiles_asset_path = self.asset_directory+'/partition_tiles'
        
        # Export the partitions
        task_1 = ee.batch.Export.table.toAsset(
            collection = partitions,
            description = 'Export-Table-Partitions', 
            assetId = partition_asset_path
            )
        
        # Export the sample points
        task_2 = ee.batch.Export.table.toAsset(
            collection = sample_points,
            description = 'Export-Table-SamplePoints', 
            assetId = sample_points_asset_path
            )
            
        # Export the sample tiles
        task_3 = ee.batch.Export.table.toAsset(
            collection = sample_tiles,
            description = 'Export-Table-SampleTiles', 
            assetId = sample_tiles_asset_path
            )
        
        #Start the tasks
        task_1.start()
        task_2.start()
        task_3.start()
        
        return None
