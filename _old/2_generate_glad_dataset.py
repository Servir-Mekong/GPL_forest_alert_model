import ee
from utils import task_monitor

ee.Initalize()

class SyntheticAlertGenerator():
    ''' 
    Notes:
    Before going hog-wild, just do the whole process for 1 points
    
    Program inputs:
    --num_days_accum_backward (int):
    --num_days_accum_forward  (int):
    --
    
    # To do:
    # Change how the nested loop work and turn it into 2 loops --> one to get a list of all of the values needed (maybe compute the label too)
    # Then run the export in a seperate step. 
    
    ''' 
    
    def __init__(self, sample_locations, projection, forward_label_fuzz, backward_label_fuzz, 
                       kernel_size, export_folder, num_sentinel_images):
        
        # Define the projection, the export location (google drive)
        self.sample_locations = sample_locations
        self.projection = projection
        self.forward_label_fuzz = forward_label_fuzz
        self.backward_label_fuzz = backward_label_fuzz
        self.kernel_size = kernel_size
        self.export_folder = export_folder
        self.sample_groups = sample_groups
        self.num_sentinel_images = num_sentinel_images
        
        # Get the bounds of the area with 
        self.study_area = self.sample_groups.geometry().bounds()
        
        # Compute the number of features 
        self.num_features = self.sample_groups.size().getInfo()
        print('Num exports:', self.num_features)
        
        # Convert the string representation of the sentinel lists into an image
        self.sample_groups = sample_groups.map(self.__id_list_to_string)
        
        # Set the kernel used for extractign the covariates
        self.kernel = self.__create_kernel(kernel_size)
        
        # Load the Sentinel 1 GRD dataset
        self.sentinel = ee.ImageCollection("COPERNICUS/S1_GRD") \
            .filterBounds(study_area) \
            .filterMetadata('orbitProperties_pass', 'equals', 'DESCENDING') \
            .filterDate('2018-10-01', '2022-12-31') \
            .select(['VV','VH'])
            
        # Load in the task monitor object
        self.monitor = task_monitor.GEETaskManager()
        
        return None
    
    ###########################################################################
 
    def aggregate_sar_for_alerts (self):

        # Get the samples
        sentinel_ids = self.__get_sentinel_ids_over_centroid()
        
        # Convert the list over each coordinate into the groups needed for creating the dataset
        sentinel_groups = ee.FeatureCollection(self.__ids_to_feature_groups(sentinel_ids)).flatten()
        
        # Run the export
        task = ee.batch.Export.table.toAsset(
            collection = sentinel_groups, 
            description = 'SERVIR-GPL-RTM-FeatureGroups', 
            assetId = 'users/JohnBKilbride/SERVIR/real_time_monitoring/glad_feature_groups_test'
            )
        task.start()
        
        return None

    # Sample the model data
    def __sample_model_data (self, stack, kernel_size, sample_point):
        '''
        Sample the different rasters
        '''      
        stack = ee.Image(stack)
        sample_point = ee.Feature(sample_point)
        
        # Define the kernel used for sampling
        kernel_cols = ee.List.repeat(1, kernel_size)
        kernel_matrix = ee.List.repeat(kernel_cols, kernel_size)
        kernel = ee.Kernel.fixed(kernel_size, kernel_size, kernel_matrix)
        
        # Select the points with the matching ID
        sample_points = ee.FeatureCollection([sample_point])
        
        # Convert the stack to an array neighborhood
        stack_array = stack.neighborhoodToArray(
            kernel = kernel, 
            defaultValue = -9999
            )
        
        # Run the sampling proceedure
        samples = stack_array.reduceRegions(
            collection = sample_points, 
            reducer = ee.Reducer.first(), 
            scale = 25, 
            tileScale = 1
            )
            
        return ee.Feature(None, {'feature_tensor': ee.Feature(samples.first())})

    # Generate the training features
    def __get_sentinel_ids_over_centroid (self):
        
        # Map over the features in the feature collection
        def map_over_sample_points (sample_point):
            
            # Cast the feature
            sample_point = ee.Feature(sample_point)
            
            # Get the start_date and the end_date for the filtering period
            map_start_date = ee.Date('2019-01-01').advance(-1*(self.num_sentinel_images-1), 'months')
            map_end_date = ee.Date('2022-01-01')
            
            # Take a sentinel image and return a feature with the metadata
            def extract_sentinel_metadata (sentinel):
                
                # Cast the sentinel image
                sentinel = ee.Image(sentinel)
                
                # Get the system:index, system:time_start, and other prioperties
                index = ee.String(sentinel.get('system:index'))
                sys_time = sentinel.date().millis()
                
                return ee.Feature(sample_point.geometry(), {'system:index': index, 'system:time_start':sys_time})
                
            images_ids = self.sentinel.filterBounds(sample_point.geometry()) \
                .filterDate(map_start_date, map_end_date) \
                .map(extract_sentinel_metadata)
            
            return ee.Feature(sample_point.geometry(), {'sentinel_info': ee.FeatureCollection(images_ids)})
            
        return self.sample_locations.map(map_over_sample_points)

    
    def __list_to_string(self, input_list):
        '''
        Convert the string to a list
        Credit: Guy Ziv
        '''
        # function to iterate over the elements of the input list of strings
        def iterate_over_elements (x,s):
            return ee.String(s).cat(ee.String(x).cat(','))
        
        nums = ee.String(input_list.iterate(iterate_over_elements,ee.String('['))).slice(0,-1).cat(']')
        
        return nums

    def __ids_to_feature_groups (self, sample_points):
        '''Convert the list of Sentinel 1 images associated with each point into groups of metrics'''  
        # Map over each of the sample points
        def points_to_feature_groups (point):
            
            # Get the previous observations required for each sentinel image
            def get_feature_groups (feature):
            
                # Cast the feature
                feature = ee.Feature(feature)
                
                # Get the filter dates
                start_date = ee.Date(feature.get('system:time_start')).advance(-24 * (self.num_sentinel_images-1), 'day')
                end_date = ee.Date(feature.get('system:time_start'))
                
                # Filter the info to get the proceeding images
                filtered_info = all_info.filterDate(start_date, end_date) \
                    .limit(self.num_sentinel_images-1, 'system:time_start', False) \
                    .toList(1e6)
                
                # Combine the metrics
                feature_ids = ee.FeatureCollection([feature]).merge(filtered_info) \
                    .sort('system:time_start', False) \
                    .aggregate_array('system:index')
                    
                # Convert the feature id's list to a string for export
                feature_strings = self.__list_to_string(feature_ids)
                
                return ee.Feature(feature.geometry(), {'ordered_sentinel_ids': feature_strings, 'system:time_start':end_date.millis()})
            
            # Cast the point
            point = ee.Feature(point)
            
            # Get the feature collection of Sentinel information
            all_info = ee.FeatureCollection(point.get('sentinel_info'))
            feature_info = all_info.filterDate('2019-01-01','2021-12-31')
            
            return feature_info.map(get_feature_groups)
        
        return sample_points.map(points_to_feature_groups)

    #####################################
        
    def generate_dataset(self):        
        
        # Load the GLAD Alerts
        glad_alerts = self.__load_formatted_alerts(study_area)
        
        # Load the Sentinel 1 GRD dataset
        sentinel = ee.ImageCollection("COPERNICUS/S1_GRD") \
            .filterBounds(study_area) \
            .filterMetadata('orbitProperties_pass', 'equals', 'DESCENDING') \
            .filterDate('2018-01-01', '2019-12-31') \
            .select(['VV','VH'])
        
        # Load the features
        feature_names = ["VV_1","VH_1","VV_2","VH_2","VV_3","VH_3"]
        
        # Loop over the features
        sample_group_list = sample_groups.toList(1e7)
        for i in range(0, self.num_sentinel_images):
        
            # Get the feature
            feature = ee.Feature(sample_group_list.get(i))
            
            # Run the sampling and export the feature export
            export_dataset_sample(feature, i, sentinel, glad_alerts, forward_label_fuzz, backward_label_fuzz, kernel, feature_names, export_folder)
            
        return None
        
    def __id_list_to_string (feature):
        '''Convert the string representation of the list into a list''' 
        # Cast the feature
        feature = ee.Feature(feature)
        
        # Get the list of strings and disucss
        id_list = ee.String(feature.get('ordered_sentinel_ids')) \
            .replace("\\[","","g").replace("\\]","","g").split(',')
        
        return feature.set('id_list', id_list)

    def __create_kernel (kernel_size):
        '''Generate the kernel'''
        kernel_cols = ee.List.repeat(1, kernel_size)
        kernel_matrix = ee.List.repeat(kernel_cols, kernel_size)
        kernel = ee.Kernel.fixed(kernel_size, kernel_size, kernel_matrix)
        return kernel

    def __load_formatted_alerts (study_area):
        '''Main logic of the script'''
        # Load in the GLAD forest alerts
        # Information: http:#glad-forest-alert.appspot.com/
        glad_alerts = ee.ImageCollection('projects/glad/alert/2019final') \
            .filterBounds(study_area)
        
        # Isolate the 2019 and 2020 alerts
        alerts_2019 = ee.Image(glad_alerts.select(['conf19','alertDate19']) \
            .map(lambda img: ee.Image(img).toInt16()) \
            .sort('system:time_start', False) \
            .first()).select(['alertDate19'])
        
        # Turn the images into a an image collection of day-to-day labels.
        alert_ts_2018 = create_dummy_alerts(2018)
        alert_ts_2019 = glad_alert_to_collection(alerts_2019, 2019, 'alertDate19')
        binary_alert_ts = ee.ImageCollection(alert_ts_2018.merge(alert_ts_2019))
        
        return binary_alert_ts

    def __glad_alert_to_collection (glad_alert, year, alert_band_name):
        '''Turn the single-image GLAD alert dates intoa  time-series of 365 binary masks'''
        # Create a list of dates
        days = ee.List.sequence(1, 365)
        
        # Map over the days to create the alert time-series
        def inner_map (day):
        
            # Cast the day as an ee.Number
            day = ee.Number(day).toInt16()
            
            # Create the date stamp
            img_date = ee.Date.fromYMD(year, 1, 1).advance(day, 'day').millis()
            
            # Get where the alerts are the 
            julian_alert = glad_alert.select(alert_band_name).eq(day).set('system:time_start', img_date).rename(['glad_alert_binary'])
            
            return julian_alert.toByte()
        
        
        return ee.ImageCollection.fromImages(days.map(inner_map))
    
    def __create_dummy_alerts (year):
        '''Create a series of all zero binary masks'''
        # Create a list of dates
        days = ee.List.sequence(1, 365)
        
        # Map over the days to create the alert time-series
        def inner_map (day):
        
            # Cast the day as an ee.Number
            day = ee.Number(day).toInt16()
            
            # Create the date stamp
            img_date = ee.Date.fromYMD(year, 1, 1).advance(day, 'day').millis()
            
            # Get where the alerts are the 
            julian_alert = ee.Image(0).set('system:time_start', img_date).rename(['glad_alert_binary'])
            
            return julian_alert.toByte()
    
        return ee.ImageCollection.fromImages(days.map(inner_map))

    def __glad_to_label (glad_alerts, alert_date, fuzz_forward, fuzz_backward):
        '''Convert GLAD alert into a binary'''
        # Fuzz the start and the end_date
        start_date_fuzz = alert_date.advance(-1 * fuzz_backward, 'day')
        end_date_fuzz = alert_date.advance(fuzz_forward, 'day')
        
        # Create the label from the glad_inputs and the fuzzed dates
        label = glad_alerts.filterDate(start_date_fuzz, end_date_fuzz).max().rename(['glad_alert'])
        
        return label

    def __sample_model_data (stack, kernel, sample_point):
        '''Sample the model data'''
        # Select the points with the matching ID
        sample_points = ee.FeatureCollection([sample_point])
        
        # Convert the stack to an array neighborhood
        stack_array = stack.neighborhoodToArray(
            kernel = kernel, 
            defaultValue = -9999
            )
        
        # Run the sampling proceedure
        samples = stack_array.reduceRegions(
            collection = sample_points, 
            reducer = ee.Reducer.first(), 
            scale = 25, 
            tileScale = 1
            )
        
        return ee.Feature(samples.first())


    def __export_dataset_sample (sample, sample_num, sentinel_images, glad_alerts, fuzz_forward, fuzz_backward, kernel, feature_names, export_folder_name):
        '''
        Function exports an individual sample to goolgle drive as a TFRecord,. Thesse can be combined as a TensorFlow Dataset
        '''
        
        # Cast the sample
        sample = ee.Feature(sample)
        
        # Get all of the info needed for export
        sample_ids = ee.List(sample.get("id_list")).getInfo()
        
        # Get the alert date
        alert_date = ee.Date(sample.get('system:time_start'))
        
        # Construct the GLAD Alert for the scene
        label = glad_to_label(glad_alerts, alert_date, fuzz_forward, fuzz_backward)
        
        # Convert the IDs to images
        scenes = []
        for i in range(0, len(sample_ids)):
        
            # Get the id from the list of ids
            scene = ee.Image("COPERNICUS/S1_GRD/"+sample_ids[i])
            
            # Append the scene to the list
            scenes = scenes.concat(scene)
            
            scenes = ee.ImageCollection.fromImages(scenes).select(["VV","VH"])
            
            # Generate the features
            features = scenes.toBands().rename(feature_names)
            
            # Stack the outputs
            labels_and_features = ee.Image.cat([features, label])
            
            # Run the sampling
            output = sample_model_data(labels_and_features, kernel, sample.geometry())
            
            # Create the export filename
            file_name = 'alert_record_'
            
            # Initiate the export
            task = ee.batch.Export.table.toDrive(
                collection =  ee.FeatureCollection([output]), 
                description =  "Export-Mekong-Test", 
                folder =  export_folder_name, 
                fileNamePrefix =  file_name+sample_num, 
                fileFormat =  "TFRecord"
                )
            task.start()
            
            # Log the info in the exporter 
        
    
        return None

if __name__ == "__main__":
    
    # Define the parameters for the gebneraor
    input_projection = ee.Projection('EPSG:32648')
    input_forward_label_fuzz = 7
    input_backward_label_fuzz = 7
    input_kernel_size = 256
    input_export_folder = 'SERVIR_alert_data'

    # Get the bounds of the area with 
    study_area = sample_groups.geometry().bounds()

    # Things that will become parametesr later
    sample_groups = ee.FeatureCollection("users/JohnBKilbride/SERVIR/real_time_monitoring/feature_groups").limit(10)
    
    # Instantiate the object
    
    # Run the sampling.



