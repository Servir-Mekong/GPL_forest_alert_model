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

        # Load the Sentinel 1 GRD dataset
        self.sentinel = ee.ImageCollection("COPERNICUS/S1_GRD") \
            .filterBounds(self.study_area) \
            .filterMetadata('orbitProperties_pass', 'equals', 'DESCENDING') \
            .filterDate('2019-01-01', '2022-12-31') \
            .select(['VV','VH', 'angle'])
        
        # Set the kernel used for extractign the covariates
        self.kernel = self.__create_kernel(kernel_size)
            
        # Load in the task monitor object
        self.export_monitor = task_monitor.GEETaskMonitor()
        
        return None
    
    ### Public methods
 
    def aggregate_sar_for_alerts (self):
    
        print("\nAggregating SAR imagery groups...")
                
        # Convert the list over each coordinate into the groups needed for creating the dataset
        sentinel_groups = self.__get_feature_groups()
        
        # # Create the export ID
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
        
        # Set the attribute of the GLAD alerts
        self.sample_id_groups = ee.FeatureCollection('users/' + self.username +'/' + self.feat_group_export_id)
        self.glad_labels = ee.FeatureCollection('users/' + self.username + '/' + self.glad_label_export_id) 
        
        # Convert the string representation of the sentinel lists into an image
        self.sample_id_groups = self.sample_id_groups.map(self.__id_list_to_string)
            
        # Loop over the features
        print('\nBeginning export process:')
        sample_group_list = self.sample_id_groups.toList(1e7).getInfo()
        num_exports = self.sample_id_groups.size().getInfo()
        for i in range(0, num_exports):
            
            print('Initating '+str(i+1)+' of '+str(num_exports))
        
            # Get the feature
            feature = ee.Feature(sample_group_list[i])
            
            # Run the sampling and export the feature export
            self.__export_dataset_sample(feature, i)
            
            # Check if exports need to be halted
            self.__check_for_monitor_capacity()
            
        
        # Run the monitoring of the exports
        self.export_monitor.monitor_tasks()
        self.export_monitor.reset_monitor()
        print('...export completed.')    
        
        return None
    
    ### Private methods
    
    def __check_for_monitor_capacity(self):
        
        # Compute the current capacity of the monitor
        capacity = self.export_monitor.get_monitor_capacity()
        print('Capacity:', capacity)
        
        # If monitor is less than 5% away from its maximum capacity then wait.
        if capacity > 0.95:
            while capacity > 0.95:
                print("...Monitor capacity at " + str(round(capacity * 100, 3)))
                sleep(60)
                self.export_monitor.check_status()
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
            scale = 10, 
            tileScale = 1
            )
            
        return ee.Feature(samples.first())

    def __get_feature_groups (self):
    
        # Define the "offset" for indexing puurposes
        offset = ee.Number(self.num_sentinel_images)
        
        # This function maps over the features in the "points" feature collection
        def process_individual_point (point):
        
            # Cast the point as an ee.Feature
            point = ee.Feature(point)
            
            # Get the properties required for the spatial stratification of the train and test sets later
            partition_x_coord = ee.Number(point.get('partition_x'))
            partition_y_coord = ee.Number(point.get('partition_y'))
            
            # Filter the sentinel data to the point location's geometry
            # and extract the ID's
            sentinel_id_list = self.sentinel.filterBounds(point.geometry()) \
                .sort('system:time_start') \
                .aggregate_array('system:index')
            
            # Get a list of indexs
            last_index = sentinel_id_list.size().subtract(offset).toInt16()
            index_list = ee.List.sequence(0, last_index)
            
            # This function over the indices ('index list') to aggregate the features
            def process_indices (current_index):
            
                # Cast the current index as a number
                current_index = ee.Number(current_index).toInt16()
                
                # Slice out the Sentinel IDs that are needed
                group_ids = sentinel_id_list.slice(current_index, current_index.add(offset).toInt16())
                
                # Define the output properties
                output_properties = {
                    'group_ids': group_ids, 
                    'num_ids': ee.Algorithms.If(group_ids.size(), group_ids.size(), 0),
                    'partition_x_coord': partition_x_coord,
                    'partition_y_coord': partition_y_coord
                    }
                
                return ee.Feature(point.geometry(), output_properties)
            
            return index_list.map(process_indices)
        
        # Map over the points to obtain the feature groups
        groups = ee.FeatureCollection(self.sample_locations.toList(1e7).map(process_individual_point).flatten())
        
        # Remove any instances where there aren't enough values
        groups = groups.filterMetadata('num_ids', 'equals', self.num_sentinel_images)
        
        # Convert the 'group_ids' field from a list to a string representation of a list
        output = groups.map(self.__convert_list_to_string)
        
        return output

    # Convert the an ee.List of strings into an ee.String
    def  __convert_list_to_string (self, feat):
    
        # Cast as a feature
        feat = ee.Feature(feat)
        
        # Get the property from the list
        id_list = ee.List(feat.get('group_ids'))
        
        # Iterate through the list and generate the string
        def  iterate_over_elements (x,s):
                return ee.String(s).cat(ee.String(x).cat(','))
        new_string = ee.String(id_list.iterate(iterate_over_elements, ee.String('['))).slice(0,-1).cat(']')
        
        # Define the output properties to select
        output_properties = ['ordered_sentinel_ids', 'partition_x_coord', 'partition_y_coord']
        
        return feat.set('ordered_sentinel_ids', new_string).select(output_properties)
        
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
        glad_alerts_2019 = ee.ImageCollection('projects/glad/alert/2019final') \
            .filterBounds(self.study_area)
        glad_alerts_2020 = ee.ImageCollection('projects/glad/alert/UpdResult') \
            .filterBounds(self.study_area)
        
        # Isolate the 2019 and 2020 alerts
        alerts_2019 = ee.Image(glad_alerts_2019.select(['alertDate19']) \
            .map(lambda img: ee.Image(img).toInt16()) \
            .sort('system:time_start', False) \
            .first())
        alerts_2020 = ee.Image(glad_alerts_2020.select(['alertDate20']) \
            .map(lambda img: ee.Image(img).toInt16()) \
            .sort('system:time_start', False) \
            .first())                  
        
        # Turn the images into a an image collection of day-to-day labels.
        alert_ts_2018 = self.__create_dummy_alerts(2018, 319, 365)
        alert_ts_2019 = self.__glad_alert_to_collection(alerts_2019, 2019, 'alertDate19')
        alert_ts_2020 = self.__glad_alert_to_collection(alerts_2020, 2020, 'alertDate20')
        binary_alert_ts = ee.ImageCollection(alert_ts_2018.merge(alert_ts_2019).merge(alert_ts_2020))
        
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
    
    def __create_dummy_alerts (self, year, start_day, end_day):
        '''Create a series of all zero binary masks'''
        # Create a list of dates
        days = ee.List.sequence(start_day, end_day)
        
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

    def __export_dataset_sample (self, sample, sample_num):
        '''
        Function exports an individual sample to goolgle drive as a TFRecord. These can be combined as a TensorFlow Dataset
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
        self.export_monitor.add_task('export_'+str(sample_num), task)
    
        return None

if __name__ == "__main__":
    
    # Define the parameters for the Generator
    input_sample_locations = ee.FeatureCollection("users/JohnBKilbride/SERVIR/real_time_monitoring/partition_points")
    input_username = 'JohnBKilbride'
    input_projection = ee.Projection('EPSG:32648')
    input_forward_label_fuzz = 7
    input_backward_label_fuzz = 3
    input_kernel_size = 256
    input_export_folder = 'SERVIR_alert_data'
    input_num_sentinel_images = 3
    input_feat_group_export_id = "SERVIR/real_time_monitoring/glad_feature_groups_test"
    input_glad_label_export_id = "SERVIR/real_time_monitoring/glad_labels"
    input_output_bands = ['VV','VH','angle']

    # Instantiate the object
    alert_generator = SyntheticAlertGenerator(input_sample_locations, input_username, input_projection, input_forward_label_fuzz, 
                                              input_backward_label_fuzz, input_kernel_size, input_export_folder, 
                                              input_num_sentinel_images, input_feat_group_export_id, input_glad_label_export_id,
                                              input_output_bands)
    
    # Aggregate the sentinel IDs needed for the second stage of processing
    # alert_generator.aggregate_sar_for_alerts()
    
    # Generate the GLAD Labels
    alert_generator.generate_glad_labels()
    
    # Export the training dataset to google drive
#    alert_generator.generate_synthetic_alert_dataset()
    
    print('\nProgram completed.')

