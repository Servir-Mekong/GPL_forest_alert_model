import ee

ee.Initialize()

def main(study_area, projection, x_cuts, y_cuts, username, output_dir, points):
    
    # Generate the partitions
    partitions = create_dataset_image(study_area, projection, x_cuts, y_cuts)
    
    # Perform the intersection
    points =  partitions.sampleRegions(
        collection = points, 
        scale = 10, 
        projection = projection,
        geometries = True
        )
    
    # Export the points to my assets
    task = ee.batch.Export.table.toAsset(
        collection = points, 
        description = 'Export-Sample-Locations-Dataset', 
        assetId = 'users/'+username+'/'+output_dir+'/'+'sample_locations_2019_2020_train_val_test_50k'
        )
    task.start()

    return None

# Contains the main logic of the script
def create_dataset_image (study_area, projection, x_cuts, y_cuts):

    # Project the geometry into the target projection
    bounding_box = study_area.bounds(ee.ErrorMargin(0.0001, 'projected'), projection) \
        .transform(projection) \
        .bounds(ee.ErrorMargin(1, 'projected'), projection)
        
    # Get the bounding box partitions
    partitions = partition_bounding_box(bounding_box, x_cuts, y_cuts, projection)
    
    # Randomly assign each partition to a set
    partitions = partitions.randomColumn()
    train = partitions.filterMetadata('random', 'less_than', 0.70)
    validation = partitions.filterMetadata('random', 'greater_than', 0.70) \
        .filterMetadata('random', 'less_than', 0.90)
    test = partitions.filterMetadata('random', 'greater_than', 0.90)
    
    # Adds the a property called "dataset_subset" to each feature
    train = add_model_set_property(train, 1)
    validation = add_model_set_property(validation, 2)
    test = add_model_set_property(test, 3)
    
    # Recombine the datasets
    combined = train.merge(validation).merge(test)
    
    # Create the binary image to sample
    dataset_image = ee.Image.constant(0).toByte().paint(combined, 'dataset_subset') \
        .rename('dataset_subset')
    
    return dataset_image

# Divide the geometry into pieces
def partition_bounding_box (box, x_cuts, y_cuts, projection): 
    
    # Get the necessary coordinates from the bounding box
    coords = ee.Array(box.coordinates())
    
    # Assign the coordinates to each corner
    xy_0_0 = ee.Geometry.Point(coords.slice(1,0,1).reshape([-1]).toList(), projection)
    xy_1_1 = ee.Geometry.Point(coords.slice(1,2,3).reshape([-1]).toList(), projection)
    
    # Get the distance between The LLH corner and the URH corner
    delta_x_y = ee.Array(xy_1_1.coordinates()).subtract(ee.Array(xy_0_0.coordinates()))
    delta_x = delta_x_y.get([0]).divide(x_cuts)
    delta_y = delta_x_y.get([1]).divide(y_cuts)
    
    # Create a feature collection of all of the verticies 
    # Loop throuhg the y coordinates in the outer loop and
    # loop through the x coordinates in the inner loop
    xy_pairs = []
    for y in range(0, y_cuts+1):
        for x in range(0, x_cuts+1):
            xy_pairs.append([x,y])
    
    # Convert the xy pairs into points
    # Note the def requires scope considerations...
    def xy_to_points (pair):
    
        # Get the x and y values
        x_coord = ee.Number(ee.List(pair).get(0))
        y_coord = ee.Number(ee.List(pair).get(1))
        
        # Translate the origin to obtain the new point
        x_translate = delta_x.multiply(x_coord).multiply(-1)
        y_translate = delta_y.multiply(y_coord).multiply(-1)
        translate_prj = projection.translate(x_translate, y_translate)
        point_geo = ee.Geometry.Point(xy_0_0.transform(translate_prj).coordinates(), projection)
        
        # Create the new feature that needs to be exported
        feature = ee.Feature(point_geo, {'grid_x':x_coord, 'grid_y':y_coord})
        
        return feature
    
    # Turn the client-side list of xy-pairs into a FeatureCollection of points
    grid_vertices = ee.FeatureCollection(ee.List(xy_pairs).map(xy_to_points))
    # Create a list of the starting vertices from which to generate the partitions
    partition_pairs = []
    partition_id = 0
    for y in range(0, y_cuts):
        for x in range(0, x_cuts):
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
        partition_geo = ee.Geometry.Polygon([[llh.coordinates(), lrh.coordinates(), urh.coordinates(), ulh.coordinates()]], projection, False)
        
        # Append the feature to the output
        partition = ee.Feature(partition_geo, {'partition_id':id_str, 'partition_x':x_coord, 'partition_y':y_coord})
        
        return partition
        
    partitions = ee.FeatureCollection(ee.List(partition_pairs).map(partitions_pairs_to_partitions))

    return partitions

# Adds a property called model set to each of the features
def add_model_set_property (features, set_name):
    def add_the_new_property (feat):
        return feat.set('dataset_subset', set_name)    
    return features.map(add_the_new_property)

#  Compute the constants needed in the process of the grid generation.
def compute_grid_constants (sample_partition, num_samples, prj):

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
    nx = calculate_sample_points_x(width, height, num_samples)
    ny = ee.Number(num_samples).divide(nx)
    
    # Round the two values
    nx = nx.round().getInfo()
    ny = ny.round().getInfo()
    
    return [width, height, nx, ny]

if __name__ == "__main__":
    
    # Script parameters
    input_points = ee.FeatureCollection('users/JohnBKilbride/SERVIR/real_time_monitoring/sample_locations_2019_2020_50k')
    input_study_area = input_points.geometry()
    input_projection = ee.Projection('EPSG:32648')
    input_x_cuts = 20
    input_y_cuts = 20
    input_username = 'JohnBKilbride'
    input_output_dir = 'SERVIR/real_time_monitoring'
    
    
    
    print('Program initiated...')
    main(input_study_area, input_projection, input_x_cuts, input_y_cuts, input_username, input_output_dir, input_points)
    print('\nProgram completed.')



