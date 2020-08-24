import ee
from time import sleep
from os import system
from utils import task_monitor

ee.Initialize()



class GenerateTrainingData():
    
    # Global variables
    __RAND_SEED = 440324767
    
    def __init__(self, path_to_sentinel, path_to_reference, num_samples_per_ref,
                 train_set_size, kernel_size, export_batch_size, gee_acc_name,
                 gdrive_folder):
    
        # Class attributes defined at instantiation
        self.__path_to_sentinel = path_to_sentinel
        self.__path_to_reference = path_to_reference
        self.__num_samples_per_ref = num_samples_per_ref
        self.__train_set_size = train_set_size
        self.__kernel_size = kernel_size
        self.__export_batch_size = export_batch_size
        self.__gee_acc_name = gee_acc_name
        self.__gdrive_folder = gdrive_folder
        
        # Load in the task monitor
        self.__task_monitor = task_monitor.GEETaskMonitor()
        
        
        # Run any error checks
        self.__check_train_size()
        
        return None
    
    ##### Public methods #####
    
    def InitiateProcessing(self):
        
        # # Generate the assets needed
        # self.__create_assets()
    
        # # Clean the reference polygon dataset and generate the sample location centroids
        # self.__perform_data_formatting()
        
        # Generate and export the model training data
        self.__generate_model_data()
        
        return None

    ##### Private functions - Higher Level  #####
    
    def __create_assets(self):
        '''Create the GEE assets needed to do the processing'''
        system('earthengine create folder users/' + self.__gee_acc_name + '/SERVIR')
        system('earthengine create folder users/' + self.__gee_acc_name + '/SERVIR/real_time_monitoring')
        return None
    
    def __perform_data_formatting(self):
        
        # Load and clean the reference polygon dataset
        reference = self.__load_reference_data()
        
        # Generate the sample locations
        sample_locations = self.__generate_sample_locations(reference)

        # Export the cleaned reference data and the sample locations
        self.__export_ref_polys_and_sample_locations(reference, sample_locations)
        
        return None
    
    
    def __load_reference_data(self):
        '''
        This function loads, cleans, and formats the refence polygon dataset.
        
        Data cleaning: We assume that the YYYMMDD data stamp is an 8 character sequence
        which can be converted into a YYYY-MM-DD date stamp. As such, all properties 
        which lack an 8 digit date stamp will be discarded.
        
        Data formatting: 
            -A system:time_stop property is added to each feature.
            -Cast the "Id" property as an integer.
        '''
        # Load the reference data
        raw_reference = ee.FeatureCollection(self.__path_to_reference)
        
        # Remove polygons with an invalid time-stamp and add a system:time_start
        # property to each feature
        valid_reference = raw_reference.map(self.__get_time_stamp_length) \
            .filterMetadata('YYYYMMDD_length', 'equals', 8) \
            .map(self.__YYYYMMDDD_to_system_time)
            
        # Cast the ID as an integer
        formatted_reference = valid_reference.map(self.__cast_feat_id_to_int) 
            
        return formatted_reference
    
    def __generate_sample_locations(self, reference):
        
#        # Randomly split into a training and test set
#        reference = reference.randomColumn('random', self.__RAND_SEED).sort('random')
#        training = reference.filterMetadata('random', 'less_than', self.__train_set_size)
#        testing = reference.filterMetadata('random', 'greater_than', self.__train_set_size)
        
        # NOTE: Until we have more reference data, we will split the dataset by date
        training = reference.filterDate(ee.Date(1548892800000).advance(-1, 'day'), ee.Date(1551052800000).advance(1, 'day'))
        testing = reference.filterDate(ee.Date(1553644800000).advance(-1, 'day'), ee.Date(1556236800000).advance(1, 'day'))
        
            
        # Get the points for each dataset
        train_points = ee.FeatureCollection(training.map(self.__sample_reference_location)
            .aggregate_array('sample_points')
            .flatten())
        test_points = ee.FeatureCollection(testing.map(self.__sample_reference_location)
            .aggregate_array('sample_points')
            .flatten())
        
        # Assign the label names
        train_points = self.__assign_train_test_label(train_points, 'Train')
        test_points = self.__assign_train_test_label(test_points, 'Test')
        
        # Merge the two point datasets
        combined_points = train_points.merge(test_points)
        
        return combined_points
    
    def __generate_model_data(self):
    
        # Load in the Ground Truth
        reference = ee.FeatureCollection('users/JohnBKilbride/SERVIR/real_time_monitoring/ref_poly_formatted')
        
        # Load in the Sentinel Imagery
        # sentinel = ee.ImageCollection('projects/cemis-camp/assets/Sentinel1') \
        #     .filterBounds(reference.geometry().bounds()) \
        #     .filterMetadata('orbitdirection', 'not_equals', 'ASCENDING') \
        #     .map(self.__gamma_nought_to_db) \
        #     .select(['VV','VH'])
        sentinel = ee.ImageCollection("COPERNICUS/S1_GRD") \
            .filterBounds(reference.geometry().bounds()) \
            .filterMetadata('orbitdirection', 'not_equals', 'ASCENDING') \
            .select(['VV','VH'])
        
        # Load in the points and seperate into the trainin and the test sets
        centroids = ee.FeatureCollection("users/JohnBKilbride/SERVIR/real_time_monitoring/sample_centroids")
        train_points = centroids.filterMetadata('Set', 'equals', 'Train')
        test_points = centroids.filterMetadata('Set', 'equals', 'Test')
        
        # Generate a study area mask to speed up subsquent computations
        study_area_mask = self.__generate_study_area_mask(train_points.merge(test_points))
        
        # Create a function that splits the training set into smaller chunks
        print('\nInitiating exporting training and testing sets...')
        self.__process_points(train_points, reference, sentinel, study_area_mask, 'Train')
        self.__process_points(test_points, reference, sentinel, study_area_mask, 'Test')
        
        # Begin Monitoring Exports
        print('\nMonitoring exports...')
        self.__task_monitor.begin_monitoring()
        print('...completed.')
       
        # Reset the task monitor
        self.__task_monitor.reset_monitor() 
        
        return None

    ##### Private functions - Lower Level #####
    
    def __check_train_size(self):
        '''Check that the input training value is a float between 0 and 1'''
        if self.__train_set_size > 1 or self.__train_set_size < 0:
            raise ValueError("'train_set_size' must be between 0 and 1 (exclusive end points).")
        return None

    def __get_time_stamp_length(self, input_feature): 
        '''
        Compute the length of the 'YYYYMMDD' property in each feature. 
        We are assuming that features where YYYYMMDD length is <8 are invalid. 
        '''
        # Cast as an ee.Feature
        input_feature = ee.Feature(input_feature)
        
        # Get the length of the YYYYMMDD property (treating it as a string)
        YYYYMMDD_length = ee.Number(input_feature.get('YYYYMMDD')).toInt32().format().length()
        
        return input_feature.set('YYYYMMDD_length', YYYYMMDD_length)
    
    def __cast_feat_id_to_int (self, feat):
        '''
        Cast the 'Id' property as an integer. For the purposes of this work flow, we always
        assume the "Id" property is an integer (not a string).
        '''
        # Cast the ee.Feature
        feat = ee.Feature(feat)
        
        # Format the id
        feat_id = ee.Number(feat.get('Id')).toInt32()
        
        return feat.set('Id', feat_id)
    
    def __YYYYMMDDD_to_system_time (self, feat): 
        '''
        Add a system:time_start property to the feature
        '''
        # Cast the ee.Feature
        feat = ee.Feature(feat)
        
        # Get the YYYYMMDD property
        ymd = ee.Number(feat.get('YYYYMMDD')).toInt32().format()
        
        # Create the date object
        year = ee.Number.parse(ymd.slice(0,4)).int16()
        month = ee.Number.parse(ymd.slice(4,6)).int16()
        day = ee.Number.parse(ymd.slice(6,8)).int16()
        date = ee.Date.fromYMD(year, month, day)
        
        return feat.set('system:time_start', date.millis())
    
    def __sample_reference_location(self, feat):
        '''
        Generate the points within a buffered bounding box around each target.
        '''
        # Cast the feature as a feature
        feat = ee.Feature(feat)
        
        # Get the ID from the feature
        feat_id = ee.Number(feat.get('Id')).toInt32()
        
        # Get the geometry from the feature
        feat_geo = feat.geometry()
        
        # Get the date from the object
        feat_date = ee.Date(feat.get('system:time_start')).millis()
        
        # Create a buffered bounding box
        sample_bounds = feat_geo.buffer(10 * (self.__kernel_size - 5), 1) 
        
        # Generate the samples
        samples = ee.Image.pixelLonLat().sample(
            region = sample_bounds, 
            scale = 10, 
            numPixels = self.__num_samples_per_ref + 3, 
            seed = self.__RAND_SEED, 
            dropNulls = False, 
            geometries = True
            ).limit(self.__num_samples_per_ref)
        
        # Add the date stamp for each 
        def inner_map (inner_feat): 
            
            # Cast as an ee.Feature
            inner_feat = ee.Feature(inner_feat)
            
            # Set the new metadata property - system:time_start and ID
            return inner_feat.set({'Id': ee.String(feat_id), 'system:time_start':feat_date})
            
        samples = ee.FeatureCollection(samples).map(inner_map)
        
        return feat.set('sample_points', samples.toList(1e3))
    
    def __assign_train_test_label(self, points, label):
        '''
        Assign a property called "Set" to each point to indicate if it is
        part of the training or the test set. 
        '''
        def inner_map(feat):
            return ee.Feature(feat).set('Set', label)
        return ee.FeatureCollection(points.map(inner_map))
    
    def __gamma_nought_to_db (self, image):
        '''Convert the gamma nought values to dB'''
        # Select the VV and VH bands
        gamma_nought = image.select(['VV', 'VH'])
        
        # Perform the conversion - 10 * log10(abs(val))
        db = ee.Image(10).multiply(gamma_nought.abs().log10()).float()
        
        return image.addBands(db, None, True).toFloat()

    def __generate_study_area_mask(self, feat_col):
        '''
        Generate a binary mask over the possible sampling areas
        '''
        # Cast the FeatureCollection
        feat_col = ee.FeatureCollection(feat_col)
        
        # Get the mask bounds from the buffered bounding box cover the input
        # FeatureCollection's geometry
        col_mask_area = feat_col.geometry().bounds().buffer(1000, 1)
        
        # Generate the binary mask
        output_mask = ee.Image.constant(0).byte().paint(col_mask_area, 1)
        
        return output_mask

    def __ref_poly_to_image (self, feat):
        '''
        Convert the reference polygon into a raster image
        '''
        # Cast the feature
        feat = ee.Feature(feat)
        
        # Get the Id from the reference polygon
        feat_id = ee.Number(feat.get('Id')).toInt32()
        
        # Get the geoemtry from the feature
        poly = feat.geometry()
        
        # Get the image date
        feat_date = ee.Date(feat.get('system:time_start')).millis()
        
        # Construct an image and paint the geometry onto it
        image = ee.Image.constant(0).byte().paint(poly, 1) \
            .set({'Id': feat_id,'system:time_start': feat_date}) \
            .rename('label')
        
        return image

    def __add_sentinel_predictors (self, labels, sentinel, image_mask):
        '''
        Add the predictors to the labels images
        '''        
        # Make a duplicate collection of the labels
        labels_dupe = labels.limit(1e6)
        
        # Load in the sentinel 1 imagery 
        def inner_map (label):
        
            # Cast the ee.Feature
            label = ee.Image(label)
            
            # Get the date from th
            label_date = label.date()
            
            # Get all of the potential labels that occured for this aquisition
            response = labels_dupe.filterDate(label_date.advance(-4, 'day'), label_date.advance(1,'Day')) \
                .max().rename(['label'])
            
            # Load in the aquisition t
            sentinel_pre = sentinel.filterDate(label_date.advance(-16, 'day'), label_date) \
                .sort('system:time_start', False) \
                .reduce(ee.Reducer.firstNonNull()) \
                .select(['VH_first', 'VV_first'],['VH','VV'])
            
            # Load in the aquistion t - 1
            sentinel_post = sentinel.filterDate(label_date, label_date.advance(16, 'day')) \
                .sort('system:time_start', False) \
                .reduce(ee.Reducer.firstNonNull()) \
                .select(['VH_first', 'VV_first'],['VH','VV'])
            
            # Apply normalization to the pre and post values
            sentinel_pre = sentinel_pre.rename(['VV_pre', 'VH_pre'])
            sentinel_post = sentinel_post.rename(['VV_post', 'VH_post'])
            
            # Combine the layers, apply the mask, and set the metadata
            output = sentinel_pre.addBands(sentinel_post).addBands(response) \
                .updateMask(image_mask) \
                .set({'Id': ee.Number(label.get('Id')).toInt32(), 'system:time_start': label_date.millis()})
            
            return output
        
        return labels.map(inner_map)

    def __sample_model_data (self, all_points, input_rasters, kernel_size):
        '''
        Sample the different rasters
        '''        
        # Define the kernel
        kernel_cols = ee.List.repeat(1, kernel_size)
        kernel_matrix = ee.List.repeat(kernel_cols, kernel_size)
        kernel = ee.Kernel.fixed(kernel_size, kernel_size, kernel_matrix)
        
        # Sample the predictors associated with each raster layer
        def inner_map (stack):
        
            # Cast stack to an ee.Image()
            stack = ee.Image(stack)
            
            # Get the ID from stack
            stack_id = ee.Number(stack.get('Id')).toInt32()
            
            # Select the points with the matching ID
            sample_points = all_points.filterMetadata('Id', 'equals', stack_id)
            
            # Convert the stack to an array neighborhood
            stack_array = stack.neighborhoodToArray(
                kernel = kernel, 
                defaultValue = 0
                )
            
            # Run the sampling proceedure
            samples = stack_array.reduceRegions(
                collection = sample_points, 
                reducer = ee.Reducer.first(), 
                scale = 10, 
                tileScale = 1
                )
            
            return ee.Feature(None, {'samples': samples.toList(1e6)})
        
        # Map the inner function over the raster stack
        input_rasters = input_rasters.map(inner_map)
        outputs = ee.FeatureCollection(input_rasters.aggregate_array('samples').flatten())
        
        return outputs
    
    def __filter_labels_by_points(self, reference_images, sample_points):
        '''
        Filter out ids not in the sample points
        '''
#        # Get the ID's from each of the sample_points
#        def str_to_int (in_str):
#            return ee.Number.parse(in_str).toInt32()
        
        ids = sample_points.aggregate_array('Id').distinct()#.map(str_to_int)
        
        # Perform the filtering
        filtered = reference_images.filter(ee.Filter.inList('Id', ids))
        
        return filtered

    def __sample_tfrecords_wrapper (self, input_points, reference, sentinel, study_area_mask):
        '''
        Run the sampling over each set of input points
        '''
        # Generate an image collection from the polygons
        labels = ee.ImageCollection(reference.map(self.__ref_poly_to_image))
        
        # Filter out the labels belonging to the other set
        filtered_labels = self.__filter_labels_by_points(labels, input_points)
        
        # Add the sentinel predictors to the labels
        model_stacks = self.__add_sentinel_predictors(filtered_labels, sentinel, study_area_mask)
        
        # Sample the stack of the predictors
        # Final parameter controls the NxN kernel size
        out_samples = self.__sample_model_data(input_points, model_stacks, 256)
        
        return out_samples

    def __process_points (self, points, reference, sentinel, study_area_mask, suffix):
        '''
        Split a larger points dataset into smaller pieces, run the sampling, then export
        '''
        # Turn the points into a list
        points_list = points.toList(1e6)
        
        # Get the number of splits needed
        num_splits = ee.Number(points.size()).divide(self.__export_batch_size).round().getInfo()
        
        # Loop through the splits
        for cur_split in range(0, num_splits+1):
                    
            # Isolate the elements to be processed
            start_index = cur_split * self.__export_batch_size 
            end_index = (cur_split+1) * self.__export_batch_size
            elements = ee.FeatureCollection(points_list.slice(start_index, end_index))
            
            # Pass the points through the sampler
            samples = self.__sample_tfrecords_wrapper(elements, reference, sentinel, study_area_mask)
            
            # Export to points to google cloud storage
            self.__export_tf_records(samples, suffix, cur_split)
        
        return None

    ##### Export functions
    def __export_ref_polys_and_sample_locations(self, reference, points):
                
        # Create the export tasks
        task1 = ee.batch.Export.table.toAsset(
            collection = reference, 
            description = 'Export-SERVIR-Ref-Poly', 
            assetId = 'users/'+self.__gee_acc_name+'/SERVIR/real_time_monitoring/ref_poly_formatted'
            )
        task2 = ee.batch.Export.table.toAsset(
            collection = points, 
            description = 'Export-SERVIR-Sample-Points', 
            assetId = 'users/'+self.__gee_acc_name+'/SERVIR/real_time_monitoring/sample_centroids'
            )
        
        # Initiate the exports
        print('\nInitating export of formatted reference polygon and sample points datasets.')
        task1.start()
        task2.start()     
    
        # Add the tasks to the task monitor
        self.__task_monitor.add_task('reference_poly', task1)
        self.__task_monitor.add_task('reference_poly', task2)
        
        # Begin Monitoring Exports
        print('\nMonitoring exports...')
        self.__task_monitor.begin_monitoring()
        print('...completed.')
       
        # Reset the task monitor
        self.__task_monitor.reset_monitor() 
    
        return None
    
    def __export_tf_records (self, export_feature, suffix, subset):
        '''
        Export the training and testing set to Google Cloud Storage
        '''
        # Define the selectors for the export
        export_selectors = ['VV_pre','VH_pre','VV_post','VH_post','label']
        
#        # Export the Training Data to Google Cloud Storage
#        task = ee.batch.Export.table.toCloudStorage(
#            collection = export_feature,
#            description = 'Export-toStorage-'+suffix+'Points',
#            bucket =  self. BUCKET_NAME,
#            fileNamePrefix = BUCKET_FOLDER + '/' + BUCKET_NAME_PREFIX + '_'+suffix+'_'+subset+'_v3',
#            fileFormat = 'TFRecord',
#            selectors = export_selectors
#            )
        
        task = ee.batch.Export.table.toDrive(
            collection= export_feature, 
            description= 'Export-toStorage-'+suffix+'Points', 
            folder=  self.__gdrive_folder, 
            fileNamePrefix = suffix+'_'+str(subset)+'_v1', 
            fileFormat  = 'TFRecord',
            selectors =  export_selectors
            )
        
        # Initiate the exports
        task.start()
        
        # Add the task to the task monitor
        self.__task_monitor.add_task(suffix+'_'+str(subset)+'_v1', task)
        
        return None
    
if __name__ == "__main__":
    
    # Define the processing parameters
    path_to_sentinel = "projects/cemis-camp/assets/Sentinel1"
    path_to_reference = "projects/cemis-camp/assets/referenceData/gplData"
    num_samples_per_ref = 35
    train_set_size = 0.7 # IGNORED FOR THE TIME BEING
    kernel_size = 256
    export_batch_size = 1024 
    gee_acc_name = "JohnBKilbride"
    gdrive_folder = 'prey_lang_data'
    
    
    # Setup and run the processing
    generator = GenerateTrainingData(path_to_sentinel, path_to_reference, num_samples_per_ref, 
                                     train_set_size, kernel_size, export_batch_size, gee_acc_name,
                                     gdrive_folder)
    
    generator.InitiateProcessing()
    
    
    
    
    
    
    
    
    