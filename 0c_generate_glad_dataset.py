import ee
from time import sleep
from utils import task_monitor

ee.Initialize()

class SyntheticAlertGenerator():

    def __init__(self, sample_locations, username, projection, forward_label_fuzz, backward_label_fuzz, 
                       kernel_size, export_folder, num_sentinel_images, feat_group_export_id,
                       glad_label_export_id, output_bands):
        
        # Define the projection, the export location (google drive)
        self.sample_locations = sample_locations
        self.username = username
        self.sample_id_groups = None
        self.glad_labels = None
        self.num_sentinel_images = num_sentinel_images
        self.forward_label_fuzz = forward_label_fuzz
        self.backward_label_fuzz = backward_label_fuzz
        self.kernel_size = kernel_size
        self.gd_export_folder_name = export_folder
        self.projection = projection
        self.feat_group_export_id = feat_group_export_id
        self.glad_label_export_id = glad_label_export_id
        self.output_bands = output_bands
        
        # Create the feature names
        self.model_feature_names = self.__create_feature_names()
        
        # Get the bounds of the area with 
        self.study_area = self.sample_locations.geometry().bounds()
        
        # Set the kernel used for extractign the covariates
        self.kernel = self.__create_kernel(kernel_size)
        
        # Load the Sentinel 1 GRD dataset
        self.sentinel = ee.ImageCollection("COPERNICUS/S1_GRD") \
            .filterBounds(self.study_area) \
            .filterMetadata('orbitProperties_pass', 'equals', 'DESCENDING') \
            .filterDate('2018-10-01', '2022-12-31') \
            .select(['VV','VH'])
            
        # Load in the task monitor object
        self.export_monitor = task_monitor.GEETaskMonitor()
        
        return None
    
    ### Public methods
 
    def aggregate_sar_for_alerts (self):
    
        print("\nAggregating SAR imagery groups...")
        
        # Get the samples
        sentinel_ids = self.__get_sentinel_ids_over_centroid()
        
        # Convert the list over each coordinate into the groups needed for creating the dataset
        sentinel_groups = ee.FeatureCollection(self.__ids_to_feature_groups(sentinel_ids)).flatten()
        
        # Create the export ID
        export_asset_id = 'users/' + self.username +'/' + self.feat_group_export_id
        
        # Run the export
        task = ee.batch.Export.table.toAsset(
        collection = sentinel_groups, 
        description = 'SERVIR-GPL-RTM-FeatureGroups', 
        assetId = export_asset_id
        )
        task.start()
        
        # Add the task to the task monitor and wait until export completes
        print('\nExporting the Sentinel ID feature groups.')
        self.export_monitor.add_task('generate_feature_groups', task)
        self.export_monitor.monitor_tasks()
        print('...export completed.')
        
        # Load and set the sample groups as a class attribute
        self.sample_id_groups = ee.FeatureCollection(export_asset_id)
        self.export_monitor.reset_monitor()
        
        return None
    
def generate_glad_labels(self):

    # Convert the string representation of the sentinel lists into an image
    self.sample_id_groups = self.sample_id_groups.map(self.__id_list_to_string)
    
    # Load the GLAD Alerts
    glad_alerts = self.__load_formatted_alerts()
    
    # Export the labels over the study area
    export_images_list = glad_alerts.toList(1e5)
    num_exports = glad_alerts.size().getInfo()
    
    # Define the export region
    export_geometry = self.study_area.buffer(self.kernel_size + 10).bounds()
    
    # Loop over the images that need to be exported
    print('\nExporting the GLAD Labels...')
    for i in range(0, num_exports):
    
        print('Initiating ' + str(i+1) + ' or ' + str(num_exports))
        
        # Get and cast the image from the list
        export_image = ee.Image(export_images_list.get(i));
        
        # Construct the export id
        export_id = 'users/' + self.username + '/' + self.glad_label_export_id + '/glad_export_'+str(i)
        
        # Export the image to google drive
        task = ee.batch.Export.image.toAsset(
            image = export_image.toByte(), 
            description = 'Export-GLAD-Label-' + str(i+1), 
            assetId =  export_id,
            region = export_geometry, 
            scale = 30, 
            maxPixels = 1e13
            )
        task.start()
        
        # Log the export to the export monitor
        self.export_monitor.add_task('glad_label_'+str(i+1), task)          
    
    # Run the monitoring of the exports
    self.export_monitor.monitor_tasks()
    self.export_monitor.reset_monitor()
    print('...export completed.')
    
    # Set the attribute of the GLAD alerts
    self.glad_labels = ee.FeatureCollection('users/' + self.username + '/' + self.glad_label_export_id) 
    
    return None
        
    def generate_synthetic_alert_dataset(self):   
        
        print('\nBeginning alert generation. ')
        
        # Convert the string representation of the sentinel lists into an image
        self.sample_id_groups = self.sample_id_groups.map(self.__id_list_to_string)
        
        # Load the Sentinel 1 GRD dataset
        sentinel = ee.ImageCollection("COPERNICUS/S1_GRD") \
            .filterBounds(self.study_area) \
            .filterMetadata('orbitProperties_pass', 'equals', 'DESCENDING') \
            .filterDate('2018-01-01', '2019-12-31') \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
            .select(self.output_bands)
            
        # Loop over the features
        print('\nBeginning export process:')
        sample_group_list = self.sample_id_groups.toList(1e7).getInfo()
        num_exports = self.sample_id_groups.size().getInfo()
        for i in range(0, num_exports):
            
            print('Initating '+str(i+1)+' of '+str(num_exports))
        
            # Get the feature
            feature = ee.Feature(sample_group_list[i])
            
            # Run the sampling and export the feature export
            self.__export_dataset_sample(feature, i, sentinel)
            
            # Check if exports need to be halted
            self.__check_for_monitor_capacity()
        
        # # Run the monitoring of the exports
        # self.export_monitor.monitor_tasks()
        # self.export_monitor.reset_monitor()
        # print('...export completed.')    
        
        return None
    
    ### Private methods
    
    def __check_for_monitor_capacity(self):
        
        # Compute the current capacity of the monitor
        capacity = self.export_monitor.get_monitor_capacity()
        
        # If monitor is less than 5% away from its maximum capacity then wait.
        if capacity > 0.98:
            while capacity > 0.98:
                print("...Monitor capacity at " + round(capacity * 100, 3))
                sleep(60)
                capacity = self.export_monitor.get_monitor_capacity()
            
        return None
    
    def __create_feature_names(self):
        feature_names = []
        for i in range(0,self.num_sentinel_images):
            for feature in self.output_bands:
                feature_names.append(feature + '_' + str(i+1))
        return feature_names
    
    def __sample_model_data (self, stack, sample_point):
        '''
        Sample the different rasters
        '''      
        stack = ee.Image(stack)
        sample_point = ee.Feature(sample_point)
        
        # Select the points with the matching ID
        sample_points = ee.FeatureCollection([sample_point])
        
        # Convert the stack to an array neighborhood
        stack_array = stack.neighborhoodToArray(
            kernel = self.kernel, 
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

    def __get_sentinel_ids_over_centroid (self):
        
        # Map over the features in the feature collection
        def map_over_sample_points (sample_point):
            
            # Cast the feature
            sample_point = ee.Feature(sample_point)
            
            # Get the properties out of the feature that we want
            grid_x = sample_point.get('grid_x')
            grid_y = sample_point.get('grid_y')
            
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
                
            # Extract the sentinel metadata (the scene IDs)
            images_ids = self.sentinel.filterBounds(sample_point.geometry()) \
                .filterDate(map_start_date, map_end_date) \
                .map(extract_sentinel_metadata)
            
            # Define the output properties
            out_properties = {
                'sentinel_info': ee.FeatureCollection(images_ids),
                'grid_x': grid_x,
                'grid_y': grid_y
                }
            
            return ee.Feature(sample_point.geometry(), out_properties)
            
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
            
            # Cast the point
            point = ee.Feature(point)
            
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
                
                # Configure the output
                output = ee.Feature(feature.geometry(), {'ordered_sentinel_ids': feature_strings, 'system:time_start':end_date.millis()}) \
                    .copyProperties(point, ['partition_x', 'partition_y', 'grid_x', 'grid_y'])
                    
                return output
            
            # Get the feature collection of Sentinel information
            all_info = ee.FeatureCollection(point.get('sentinel_info'))
            feature_info = all_info.filterDate('2019-01-01','2021-12-31')
            
            return feature_info.map(get_feature_groups)
        
        return sample_points.map(points_to_feature_groups)
        
    def __id_list_to_string (self, feature):
        '''Convert the string representation of the list into a list''' 
        # Cast the feature
        feature = ee.Feature(feature)
        
        # Get the list of strings and disucss
        id_list = ee.String(feature.get('ordered_sentinel_ids')) \
            .replace("\\[","","g").replace("\\]","","g").split(',')
        
        return feature.set('id_list', id_list)

    def __create_kernel (self, kernel_size):
        '''Generate the kernel'''
        kernel_cols = ee.List.repeat(1, kernel_size)
        kernel_matrix = ee.List.repeat(kernel_cols, kernel_size)
        kernel = ee.Kernel.fixed(kernel_size, kernel_size, kernel_matrix)
        return kernel

    def __load_formatted_alerts (self):
        '''Main logic of the script'''
        # Load in the GLAD forest alerts
        # Information: http:#glad-forest-alert.appspot.com/
        glad_alerts = ee.ImageCollection('projects/glad/alert/2019final') \
            .filterBounds(self.study_area)
        
        # Isolate the 2019 and 2020 alerts
        alerts_2019 = ee.Image(glad_alerts.select(['conf19','alertDate19']) \
            .map(lambda img: ee.Image(img).toInt16()) \
            .sort('system:time_start', False) \
            .first()).select(['alertDate19'])
        
        # Turn the images into a an image collection of day-to-day labels.
        alert_ts_2018 = self.__create_dummy_alerts(2018)
        alert_ts_2019 = self.__glad_alert_to_collection(alerts_2019, 2019, 'alertDate19')
        binary_alert_ts = ee.ImageCollection(alert_ts_2018.merge(alert_ts_2019))
        
        return binary_alert_ts

    def __glad_alert_to_collection (self, glad_alert, year, alert_band_name):
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
            julian_alert = glad_alert.select(alert_band_name).gte(day.subtract(self.backward_label_fuzz)) \
                .And(glad_alert.select(alert_band_name).lte(day.add(self.forward_label_fuzz))) \
                .set('system:time_start', img_date).rename(['glad_alert_binary'])
            
            return julian_alert.toByte()
        
        return ee.ImageCollection.fromImages(days.map(inner_map))
    
    def __create_dummy_alerts (self, year):
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

    def __retrieve_label (self, alert_date):
        '''Retrieves a single glad alert from the collection'''
        # Fuzz the start and the end_date
        start_date_fuzz = alert_date
        end_date_fuzz = alert_date.advance(1, 'day')
        
        # Get the label from the set of labels
        label = self.glad_labels.filterDate(start_date_fuzz, end_date_fuzz).first()
        
        return ee.Image(label).rename(['glad_alert'])

    def __export_dataset_sample (self, sample, sample_num, sentinel_images):
        '''
        Function exports an individual sample to goolgle drive as a TFRecord. Thesse can be combined as a TensorFlow Dataset
        '''
        # Cast the sample
        sample = ee.Feature(sample)
        
        # Get the partition coordinates from the sample
        grid_x = str(ee.Number(sample.get('grid_x')).toInt16().getInfo())
        grid_y = str(ee.Number(sample.get('grid_y')).toInt16().getInfo())
        
        # Get all of the info needed for export
        sample_ids = ee.List(sample.get("id_list")).getInfo()
        
        # Get rid of the leading characters introduced by previous processing steps
        sample_ids_trimmed = []
        for sentinel_id in sample_ids:
            sample_ids_trimmed.append(sentinel_id[2:])
        
        # Get the alert date
        alert_date = ee.Date(sample.get('system:time_start'))
        
        # Construct the GLAD Alert for the scene
        label = self.__retrieve_label(alert_date)
        
        # Convert the IDs to images
        scenes = []
        for i in range(0, len(sample_ids_trimmed)):
        
            # Get the id from the list of ids
            scene = ee.Image("COPERNICUS/S1_GRD/"+sample_ids_trimmed[i])
            
            # Append the scene to the list
            scenes.append(scene)
            
        scenes = ee.ImageCollection.fromImages(scenes).select(self.output_bands)
        
        # Generate the features
        features = scenes.toBands().rename(self.model_feature_names)
        
        # Stack the outputs
        labels_and_features = ee.Image.cat([features, label])
        
        # Run the sampling
        output = self.__sample_model_data(labels_and_features, sample.geometry())
        
        # Create the export filename
        file_name = 'alert_record_' + grid_x + '_' + grid_y + '_' + str(sample_num)
        
        # Initiate the export
        task = ee.batch.Export.table.toDrive(
            collection =  ee.FeatureCollection([output]), 
            description =  "Export-Mekong-Alert-" + str(sample_num), 
            folder =  self.gd_export_folder_name, 
            fileNamePrefix =  file_name, 
            fileFormat =  "TFRecord"
            )
        task.start()
        
        # Log the info in the exporter
        self.export_monitor.add_task('export_'+str(i), task)
    
        return None

if __name__ == "__main__":
    
    # Define the parameters for the Generator
    input_sample_locations = ee.FeatureCollection("users/JohnBKilbride/SERVIR/real_time_monitoring/partition_points")
    input_username = 'JohnBKilbride'
    input_projection = ee.Projection('EPSG:32648')
    input_forward_label_fuzz = 7
    input_backward_label_fuzz = 7
    input_kernel_size = 256
    input_export_folder = 'SERVIR_alert_data'
    input_num_sentinel_images = 3
    input_feat_group_export_id = "SERVIR/real_time_monitoring/glad_feature_groups"
    input_glad_label_export_id = "SERVIR/real_time_monitoring/glad_labels"
    input_output_bands = ['VV','VH','angle']

    # Instantiate the object
    alert_generator = SyntheticAlertGenerator(input_sample_locations, input_username, input_projection, input_forward_label_fuzz, 
                                              input_backward_label_fuzz, input_kernel_size, input_export_folder, 
                                              input_num_sentinel_images, input_feat_group_export_id, input_glad_label_export_id,
                                              input_output_bands)

    # Aggregate the sentinel IDs needed for the second stage of processing
    alert_generator.aggregate_sar_for_alerts()
    
    # Generate the GLAD Labels
    alert_generator.generate_glad_labels()
    
    # Export the training dataset to google drive
    alert_generator.generate_synthetic_alert_dataset()
    
    print('\nProgram completed.')

