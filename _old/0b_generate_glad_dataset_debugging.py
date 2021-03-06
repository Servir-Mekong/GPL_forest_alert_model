import ee
from time import sleep
from utils import task_monitor
from datetime import datetime

ee.Initialize()

class SyntheticAlertGenerator():

    def __init__(self, sample_locations, partitions, username, projection, forward_label_fuzz, backward_label_fuzz, 
                       kernel_size, gcs_bucket, gcs_export_folder, num_sentinel_images, feat_group_export_id,
                       glad_label_export_id, output_bands):
        '''
        Parameters
        ----------
        sample_locations : ee.FeatureCollection
            Contains point locations over which points will be generated. This script assumes
            each ee.Feature contains a point geometry and has a property "partition_id" which
            indicates which sample partition the point is located in
        username : string
            The user's Google Earth Engine user account name
        projection : ee.Projection
            A projected coordinate system appropreate for the use's study area. Units are assumed
            to be meters.
        forward_label_fuzz : integer
            For a given alert date (e.g., 185), the number of GLAD dates prior to the current 
            alert date to include in the reference label.
        backward_label_fuzz : integer
            For a given alert date (e.g., 185), the number of GLAD dates after to the current 
            alert date to include in the reference label..
        kernel_size : integer
            The size of the exported tensor for training a semantic segmentation model (e.g., 256)
        export_folder : string
            DESCRIPTION.
        num_sentinel_images : integer
            The number of Sentinel-1 images to include in the exported feature tensor.
        feat_group_export_id : string
            DESCRIPTION.
        glad_label_export_id : string
            DESCRIPTION.
        output_bands : list
            A list of strings indicating which Sentinel-1 bands should be included in 
            the exported feature tensors. Valid options: VV, VH, angle

        Returns
        -------
        None.

        '''
        self.sample_locations = sample_locations
        self.partitions = partitions
        self.username = username
        self.sample_id_groups = None
        self.glad_labels = None
        self.num_sentinel_images = num_sentinel_images
        self.forward_label_fuzz = forward_label_fuzz
        self.backward_label_fuzz = backward_label_fuzz
        self.kernel_size = kernel_size
        self.gcs_bucket = gcs_bucket
        self.gcs_export_folder = gcs_export_folder
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
            .filterDate('2019-01-01', '2022-12-31') \
            .select(['VV','VH', 'angle'])
        
        # Set the kernel used for extractign the covariates
        self.kernel = self.__create_kernel()
            
        # Load in the task monitor object
        self.export_monitor = task_monitor.GEETaskMonitor()
        
        # The partition ids for sampels in the training, validation, and test set
        # NOTE: these are assigned during the function call to the aggregate_sar_for_alerts
        # function
        self.train_set_ids = None
        self.val_set_ids = None
        self.test_set_ids = None
        
        return None
    
    ### Public methods
 
    def aggregate_sar_for_alerts (self):
        """
        This script aggregates the groups of ID's needed to construct each of the alerts
        which will be exported. This results in a ee.FeatureCollection being exported to 
        the user's Google Earth Engine account -- this step is performed before exporting
        the feature tensors to avoid hiting compute limits. 

        Returns
        -------
        None.

        """
        print("\nAggregating SAR imagery groups...")

        # Assign each of the partitions to the train, validation, or test sets
        # The labeled_sample_points are used for the final test dataset
        # The labeled_sample_partitions are used to generate the final training and validation datasets
        labeled_sample_points, labeled_sample_partitions = self.__assign_sample_to_set()
        
        # Perform "targeted sampling" in the training and validation partitions
        sentinel_groups_partitions = self.__get_feature_groups_paritions(labeled_sample_points)
        
        # # Convert the list over each coordinate into the groups needed for creating the dataset
        # sentinel_groups_points = self.__get_feature_groups_points(labeled_sample_points)
                        
        # # Create the export ID
        # export_asset_id = 'users/' + self.username +'/' + self.feat_group_export_id
        
        # # Run the export
        # task = ee.batch.Export.table.toAsset(
        #     collection = sentinel_groups, 
        #     description = 'SERVIR-GPL-RTM-FeatureGroups', 
        #     assetId = export_asset_id
        #     )
        # task.start()
        
        # # Add the task to the task monitor and wait until export completes
        # print('\nExporting the Sentinel ID feature groups.')
        # self.export_monitor.add_task('generate_feature_groups', task)
        # self.export_monitor.monitor_tasks()
        # print('...export completed.')
        
        # # Load and set the sample groups as a class attribute
        # self.sample_id_groups = ee.FeatureCollection(export_asset_id)
        # self.export_monitor.reset_monitor()
        
        return None
    
    def generate_glad_labels(self):
        '''
        The UMD GLAD alerts are processed into a time-series of binary disturbance / 
        no-disturbance images which can later be queried. Pre-computing these lables
        expidates the export process later. Each label is constructed by combining all
        alert dates for each day from 2019-2020. For a particular date, this aggregation
        includes alerts produced forward_label_fuzz days after and backward_label_fuzz 
        days prior to a given alert date (e.g., 185 - 2019).

        Returns
        -------
        None.

        '''
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
        '''
        Controls the main logic for exporting the alert records to Google Cloud Storage.

        Returns
        -------
        None.

        '''
        print('\nBeginning alert generation. ')
        
        # Set the attribute of the GLAD alerts
        self.sample_id_groups = ee.FeatureCollection('users/' + self.username +'/' + self.feat_group_export_id) \
            .randomColumn(columnName='random', seed=835791) \
            .sort('random').limit(1)
        self.glad_labels = ee.FeatureCollection('users/' + self.username + '/' + self.glad_label_export_id) 
        
        # Convert the string representation of the sentinel lists into an image
        self.sample_id_groups = self.sample_id_groups.map(self.__string_to_id_list)
            
        # Loop over the features
        print('\nBeginning export process:')
        sample_group_list = self.sample_id_groups.toList(1e7).getInfo()
        num_exports = self.sample_id_groups.size().getInfo()
        for i in range(0, num_exports):
        
            # Get the feature
            feature = ee.Feature(sample_group_list[i])
            
            # Run the sampling and export the feature export
            self.__export_dataset_sample(feature, i, num_exports)
            
            # Check for completed exports every 250 iterations
            if  i % 250 == 0:
                self.__check_for_monitor_capacity()
            
        # Run the monitoring of the exports
        self.export_monitor.monitor_tasks()
        self.export_monitor.reset_monitor()
        print('...export completed.')    
        
        return None
    
    ### Private methods
    
    def __check_for_monitor_capacity(self):
        """
        Checks the capacity of the task_monitor object. If the capacity is approaching
        the limit (definined when instantiating the task_monitor), then the function
        will wait until a sufficent number of tasks have been completed (or failed).

        Returns
        -------
        None.

        """
        # Compute the current capacity of the monitor
        capacity = self.export_monitor.get_monitor_capacity()
        
        # If monitor is less than 5% away from its maximum capacity then wait.
        if capacity > 0.95:
            while capacity > 0.95:
                print("...Monitor capacity at " + str(round(capacity * 100, 3)))
                sleep(600)
                self.export_monitor.check_status()
                capacity = self.export_monitor.get_monitor_capacity()
            
        return None
    
    def __create_feature_names(self):
        """
        Helper function which formats the names of each of the features that will be exported.
        Example, if self.num_sentinel_images == 3 and self.output_bands == ['VV'] then the feature names
        will return ['VV_1', 'VV_2', 'VV_3'].

        Returns
        -------
        feature_names : TYPE
            DESCRIPTION.

        """
        feature_names = []
        for i in range(0,self.num_sentinel_images):
            for feature in self.output_bands:
                feature_names.append(feature + '_' + str(i+1))
        return feature_names
    
    def __sample_model_data (self, stack, sample_point):
        """
        Extracts the final feature tensor from the input stack. The stack is assumed to contain
        the Sentinel-1 SAR images and the GLAD Alert binary label. The dimensions of the extracted
        tensor are determiend the by the input kernel_size parameter - defined upon the object's
        instantiation. 

        Parameters
        ----------
        stack : ee.Image
            An image which contains an arbitrary number of Sentinel-1 SAR images
            and a binary GLAD alert label.
        sample_point : ee.Feature
            A feature's who's geometry is assumed to be a ee.Geometry.Point which defines
            the centroid of the extracted feature.

        Returns
        -------
        ee.Feature
            A feature which contains a tensor which can be exported as a TFRecord. 

        """
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
   
    def __assign_sample_to_set(self):
        """
        Using the 'partition_id' field which each feature in the input partitions
        ee.FeatureCollection contains, the ids are randomly assigned to either the
        training, validation, or testing sets. These are set as class attributes 
        (ee.Lists which contain the IDs for the set)

        Returns
        -------
        None.

        """
        shuffled_partitions = self.partitions.randomColumn().sort('random').toList(1e6)
        
        # Come up with the indicies to sort the list
        num_partitions = shuffled_partitions.length()
        
        # Compute the size of the train, validation, test splits (shoot for 70-20-10)
        num_train_partitions = num_partitions.multiply(0.70).round()
        num_val_partitions = num_partitions.multiply(0.20).round()
        num_test_partitions = num_partitions.subtract(num_train_partitions).subtract(num_val_partitions)
        
        # Get the partitions ids
        val_last_index = num_train_partitions.add(num_val_partitions)
        test_first_index = num_train_partitions.add(num_val_partitions)
        test_last_index = num_train_partitions.add(num_val_partitions).add(num_test_partitions)
        
        train_set_ids = ee.FeatureCollection(shuffled_partitions.slice(0, num_train_partitions)) \
            .aggregate_array('partition_id')
        val_set_ids = ee.FeatureCollection(shuffled_partitions.slice(num_train_partitions, val_last_index)) \
            .aggregate_array('partition_id')
        test_set_ids = ee.FeatureCollection(shuffled_partitions.slice(test_first_index, test_last_index)) \
            .aggregate_array('partition_id')
    
        # Assign each sample to one of the partitions
        test_points = self.sample_locations.filter(ee.Filter.inList('partition_id', test_set_ids))
        
        # Recombine all of the points into a single set
        test_points = self.__add_model_set_labels(test_points, "test")
        
        # Recombine the points into a single dataset
        # Note: Due to the sparsity of disturbances -- we've decided to only use the grid sampling stratedgy
        # for points in the test set
        labeled_points = ee.FeatureCollection(test_points)
        
        # Isolate and label the training and validation partitions
        train_partitions = self.partitions.filter(ee.Filter.inList('partition_id', train_set_ids))
        train_partitions = self.__add_model_set_labels(train_partitions, "train")
        val_partitions = self.partitions.filter(ee.Filter.inList('partition_id', val_set_ids))
        val_partitions = self.__add_model_set_labels(val_partitions, "validation")
        
        # Combine the labeled partitions
        labeled_partitions = ee.FeatureCollection(train_partitions.merge(val_partitions))
        
        return labeled_points, labeled_partitions
    
    def __add_model_set_labels(self, input_features, model_label):
        """
        Assigns a property called 'model_set' to each ee.Feature in the input 
        ee.FeatureCollection.

        Parameters
        ----------
        input_features : ee.FeatureCollection
            A feature collection of points which will be assigned a lable: train,
            validation, or test.
        model_label : string
            A string to label each point with -- 'train', 'validation', or 'test'

        Returns
        -------
        ee.FeatureCollection
            The same ee.FeatureCopllection as the inpuit_points but each feature
            has been assined a property with the 'model_label'.

        """
        def inner_map (point):
            return ee.Feature(point).set('model_set', model_label)
        return input_features.map(inner_map) 

    def __get_feature_groups_partitions(self, input_sample_partitions):
        """
        Main logic for aggreagating the feature groups. Here, a feature group 
        describes the list of Sentinel-1 IDs which will be used to create a feature
        tensor used by a semantic segmentation algorithm. 

        Returns
        -------
        ee.FeatureCollection
            A collection where each feature contains two properties: 1. a string
            representation of an ee.List which contains the Sentinl-1 system:id
            values needed to create a particular record in the alert dataset and 2.
            the partition_id of the sample (which is needed for spatial stratification).

        """
        
        # Load the GLAD Alerts for 2019 and 2020
        glad_2019, glad_2020 = self.__load_glad_alerts_for_partition()
     
        # Iterate over each of the paritions
        def iterate_over_paritions (partition) :
            
            # Cast the partition
            partition = ee.Feature(partition)
            
            # Extract several properties from the partition feature
            partition_geometry = partition.geometry()
            partition_model_id = ee.String(partition.get('model_set'))
            
            # Run a stratified random sample of the GLAD alerts using the partition geometry
            samples_2019 = glad_2019.stratifiedSample(
                numPoints = 1, 
                classBand = "alert_sampling", 
                region = partition_geometry, 
                scale = 10, 
                classValues = [0, 1], 
                classPoints = [0, 100], 
                dropNulls = True, 
                tileScale = 4, 
                geometries = True
                )
            samples_2020 = glad_2020.stratifiedSample(
                numPoints = 1, 
                classBand = "alert_sampling", 
                region = partition_geometry, 
                scale = 10, 
                classValues = [0, 1], 
                classPoints = [0, 100], 
                dropNulls = True, 
                tileScale = 4, 
                geometries = True
                )
        
            #
            
            return 
        
        return None

    def __load_glad_alerts_for_partition (self):
        '''
        Loads the GLAD alerts required for the study area. Values of the returned images
        ranges between 1 - 365 (which corresponds to the julian day of the first instance of the detected alert)

        Returns
        -------
        alerts_2019 : ee.Image
            DESCRIPTION.
        alerts_2020 : ee.Image
            DESCRIPTION.

        '''
        # Load in the GLAD forest alerts
        # Information: http:#glad-forest-alert.appspot.com/
        glad_alerts_2019 = ee.ImageCollection('projects/glad/alert/2019final').filterBounds(self.study_area)
        glad_alerts_2020 = ee.ImageCollection('projects/glad/alert/UpdResult').filterBounds(self.study_area)
        
        # Isolate the 2019 and 2020 alerts
        alerts_2019 = ee.Image(glad_alerts_2019.select(['alertDate19']) \
            .sort('system:time_start', False) \
            .rename(['julian_day']) \
            .first()).toInt16()
        alerts_2020 = ee.Image(glad_alerts_2020.select(['alertDate20']) \
            .sort('system:time_start', False) \
            .rename(['julian_day']) \
            .first()).toInt16()       
        
        # Compute and add a binary band to each of the images
        alerts_2019_binary = alerts_2019.gt(0).rename('alert_sampling')
        alerts_2020_binary = alerts_2020.gt(0).rename('alert_sampling')
        alerts_2019 = alerts_2019.addBands(alerts_2019_binary)
        alerts_2020 = alerts_2020.addBands(alerts_2020_binary)
            
        return alerts_2019, alerts_2020
    
    def __add_year_partition_samples(self):
        '''
        

        Returns
        -------
        None.

        '''
        return None

    def __get_feature_groups_partitions(self, input_sample_partitions):
        """
        Main logic for aggreagating the feature groups. Here, a feature group 
        describes the list of Sentinel-1 IDs which will be used to create a feature
        tensor used by a semantic segmentation algorithm. 

        Returns
        -------
        ee.FeatureCollection
            A collection where each feature contains two properties: 1. a string
            representation of an ee.List which contains the Sentinl-1 system:id
            values needed to create a particular record in the alert dataset and 2.
            the partition_id of the sample (which is needed for spatial stratification).

        """
        
        # Load the GLAD Alerts for 2019 and 2020
        glad_2019, glad_2020 = self.__load_glad_alerts_for_partition()
     
        # Iterate over each of the paritions
        def iterate_over_paritions (partition) :
            
            # Cast the partition
            partition = ee.Feature(partition)
            
            # Extract several properties from the partition feature
            partition_geometry = partition.geometry()
            partition_model_id = ee.String(partition.get('model_set'))
            
            # Run a stratified random sample of the GLAD alerts using the partition geometry
            samples_2019 = glad_2019.stratifiedSample(
                numPoints = 1, 
                classBand = "alert_sampling", 
                region = partition_geometry, 
                scale = 10, 
                classValues = [0, 1], 
                classPoints = [0, 5000], 
                dropNulls = True, 
                tileScale = 4, 
                geometries = True
                )
            samples_2020 = glad_2020.stratifiedSample(
                numPoints = 1, 
                classBand = "alert_sampling", 
                region = partition_geometry, 
                scale = 10, 
                classValues = [0, 1], 
                classPoints = [0, 5000], 
                dropNulls = True, 
                tileScale = 4, 
                geometries = True
                )
        
            #
            
            return 
        
        return None

    def __load_glad_alerts_for_partition (self):
        '''
        Loads the GLAD alerts required for the study area. Values of the returned images
        ranges between 1 - 365 (which corresponds to the julian day of the first instance of the detected alert)

        Returns
        -------
        alerts_2019 : ee.Image
            DESCRIPTION.
        alerts_2020 : ee.Image
            DESCRIPTION.

        '''
        # Load in the GLAD forest alerts
        # Information: http:#glad-forest-alert.appspot.com/
        glad_alerts_2019 = ee.ImageCollection('projects/glad/alert/2019final').filterBounds(self.study_area)
        glad_alerts_2020 = ee.ImageCollection('projects/glad/alert/UpdResult').filterBounds(self.study_area)
        
        # Isolate the 2019 and 2020 alerts
        alerts_2019 = ee.Image(glad_alerts_2019.select(['alertDate19']) \
            .sort('system:time_start', False) \
            .rename(['julian_day']) \
            .first()).toInt16()
        alerts_2020 = ee.Image(glad_alerts_2020.select(['alertDate20']) \
            .sort('system:time_start', False) \
            .rename(['julian_day']) \
            .first()).toInt16()       
        
        # Compute and add a binary band to each of the images
        alerts_2019_binary = alerts_2019.gt(0).rename('alert_sampling')
        alerts_2020_binary = alerts_2020.gt(0).rename('alert_sampling')
        alerts_2019 = alerts_2019.addBands(alerts_2019_binary)
        alerts_2020 = alerts_2020.addBands(alerts_2020_binary)
            
        return alerts_2019, alerts_2020
    
    def __add_year_partition_samples(self):
        '''
        

        Returns
        -------
        None.

        '''
        return None
    
    def __get_feature_groups_points (self, input_sample_locations):
        """
        Main logic for aggreagating the feature groups. Here, a feature group 
        describes the list of Sentinel-1 IDs which will be used to create a feature
        tensor used by a semantic segmentation algorithm. 

        Returns
        -------
        ee.FeatureCollection
            A collection where each feature contains two properties: 1. a string
            representation of an ee.List which contains the Sentinl-1 system:id
            values needed to create a particular record in the alert dataset and 2.
            the partition_id of the sample (which is needed for spatial stratification).

        """
        # Define the "offset" for indexing puurposes
        offset = ee.Number(self.num_sentinel_images)
        
        # This function maps over the features in the "points" feature collection
        def process_individual_point (point):
        
            # Cast the point as an ee.Feature
            point = ee.Feature(point)
            
            # Get the properties required for the spatial stratification of the train and test sets later
            partition_id = point.get('partition_id')
            model_set = point.get('model_set')
            
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
                    'partition_id': partition_id,
                    'model_set': model_set
                    }
                
                return ee.Feature(point.geometry(), output_properties)
            
            return index_list.map(process_indices)
        
        # Map over the points to obtain the feature groups
        groups = ee.FeatureCollection(input_sample_locations.toList(1e7).map(process_individual_point).flatten())
        
        # Remove any instances where there aren't enough values
        groups = groups.filterMetadata('num_ids', 'equals', self.num_sentinel_images)
        
        # Convert the 'group_ids' field from a list to a string representation of a list
        output = groups.map(self.__convert_list_to_string)
        
        return output

    def  __convert_list_to_string (self, feat):
        """
        Converts an ee.List to an ee.String so that the features can later be 
        exported as an ee.FeatureCollection to the user's Google Earth Engine
        Assets. 

        Parameters
        ----------
        feat : ee.Feature
            The current feature that needs to be processed. Is assumed to have a 
            property called 'group_ids' which contains an ee.List which contains 
            the system:index associated with a particular Sentinel-1 scene. 

        Returns
        -------
        ee.Feature
            An feature which contains an ee.List of the Sentinel GRD system:id values.

        """
        # Cast as a feature
        feat = ee.Feature(feat)
        
        # Get the property from the list
        id_list = ee.List(feat.get('group_ids'))
        
        # Iterate through the list and generate the string
        def  iterate_over_elements (x,s):
                return ee.String(s).cat(ee.String(x).cat(','))
        new_string = ee.String(id_list.iterate(iterate_over_elements, ee.String('['))).slice(0,-1).cat(']')
        
        # Define the output properties to select
        output_properties = ['ordered_sentinel_ids', 'partition_id', 'model_set']
        
        return feat.set('ordered_sentinel_ids', new_string).select(output_properties)
        
    def __string_to_id_list (self, feature):
        """
        Converts a ee.String representation of a list into an ee.List()

        Parameters
        ----------
        feature : ee.Feature
            A feature collection containing a property called 'ordered_sentinel_ids'
            which is an ee.String representation of an ee.List containing Sentinel-1
            GRD system:index values. 

        Returns
        -------
        ee.Feature
            An ee.Feature with a new property called 'id_list which '.

        """
        # Cast the feature
        feature = ee.Feature(feature)
        
        # Get the list of strings and disucss
        id_list = ee.String(feature.get('ordered_sentinel_ids')) \
            .replace("\\[","","g").replace("\\]","","g").split(',')
        
        return feature.set('id_list', id_list)

    def __create_kernel (self):
        """
        Generates the kernel that will later be used with the .toNeighborhoods()
        function from the Google Earth Engine API. This is needed for sampling the 
        feature tensors. 

        Returns
        -------
        kernel : ee.Kernel
            A GEE Kernel of the input dimensions. 

        """
        kernel_cols = ee.List.repeat(1, self.kernel_size)
        kernel_matrix = ee.List.repeat(kernel_cols, self.kernel_size)
        kernel = ee.Kernel.fixed(self.kernel_size, self.kernel_size, kernel_matrix)
        return kernel

    def __load_formatted_alerts (self):
        """
        Controls the main logic for aggregating the UMD GLAD alerts into a 
        time-series of binary alert images. 

        Returns
        -------
        binary_alert_ts : ee.ImageCollection
            An image collection consisting of binary alerts for 2019 and 2020.

        """
        # Load in the GLAD forest alerts
        # Information: http:#glad-forest-alert.appspot.com/
        glad_alerts_2019 = ee.ImageCollection('projects/glad/alert/2019final') \
            .filterBounds(self.study_area)
        glad_alerts_2020 = ee.ImageCollection('projects/glad/alert/UpdResult') \
            .filterBounds(self.study_area)
        
        # Isolate the 2019 and 2020 alerts
        alerts_2019 = ee.Image(glad_alerts_2019.select(['alertDate19']) \
            .sort('system:time_start', False) \
            .first()).toInt16()
        alerts_2020 = ee.Image(glad_alerts_2020.select(['alertDate20']) \
            .sort('system:time_start', False) \
            .first()).toInt16()                  
        
        # Scale the julian dates of the second year by 365
        alerts_2020 = alerts_2020.add(alerts_2020.neq(0).multiply(365))
        
        # Combine the two glad alerts
        combined_alerts = ee.Image.cat([alerts_2019, alerts_2020])
            
        # Turn the images into a an image collection of day-to-day labels.
        binary_alert_ts = self.__glad_alert_to_collection(combined_alerts, 2019, 'alertDate19')
        
        return ee.ImageCollection(binary_alert_ts)

    def __glad_alert_to_collection (self, combined_alerts):
        """
        Process a single UMD GLAD alert image into a time-series of binary 
        disturbance / no-disturbance images. 

        Parameters
        ----------
        combined_alerts : ee.Image
            A GLAD alert image for the area of interest. 

        Returns
        -------
        ee.ImageCollection
            An collection of binary disturbance/no-disturbance images

        """
        # Create a list of dates
        days = ee.List.sequence(0, 729)
        
        # Map over the days to create the alert time-series
        def  inner_map (day):
            
            # Cast the day as an ee.Number
            day = ee.Number(day).toInt16()
            
            # Create the date stamp
            img_date = ee.Date.fromYMD(2019, 1, 1).advance(day, 'day').millis()
            
            # Get where the alerts are the 
            start = ee.Number.clamp(day.subtract(10), 1, 730)
            stop = ee.Number.clamp(day.add(3), 1, 730)
            julian_alert = combined_alerts.gte(start).And(combined_alerts.lte(stop)) \
                .reduce(ee.Reducer.sum()).gte(1) \
                .set('system:time_start', img_date).rename(['glad_alert_binary'])
                
            return julian_alert.toByte()
            
        return ee.ImageCollection.fromImages(days.map(inner_map))
                
    def __retrieve_label (self, alert_date):
        """
        A helper function which extracts a particular alert (using the input date)
        from an ee.ImageCollection of binary alerts. 
        
        Parameters
        ----------
        alert_date : ee.Date
            The date of the alert which needs to be retrieved

        Returns
        -------
        ee.Image
            The binary disturbance/no disturbance image retrieved from the collection.

        """
        # Fuzz the start and the end_date
        start_date_fuzz = alert_date
        end_date_fuzz = alert_date.advance(1, 'day')
        
        # Get the label from the set of labels
        label = self.glad_labels.filterDate(start_date_fuzz, end_date_fuzz).first()
        
        return ee.Image(label).rename(['glad_alert'])

    def __export_dataset_sample (self, sample, sample_num, num_exports):
        """
        Function exports an individual sample to google drive as a TFRecord. 
        These can be combined as a TensorFlow Dataset.

        Parameters
        ----------
        sample : ee.Feature
            A feature containing the following propertioes: 
                partition_id: a numerical ID indicating the sample's partition
                id_list: get 
        sample_num : integer
            A number which is appended to the file path to identify a sample uniquely.
        num_exports: integer
            The total number of exports. This is used when printing info about the export status

        Returns
        -------
        None.

        """
        # Cast the sample
        sample = ee.Feature(sample)
        
        # Get the partition coordinates from the sample
        partition_id = str(ee.Number(sample.get('partition_id')).toInt16().getInfo())
                
        # Get all of the info needed for export
        sample_ids = ee.List(sample.get("id_list")).getInfo()
        model_set = ee.String(sample.get("model_set")).getInfo()
        
        # Get rid of the leading characters introduced by previous processing steps
        sample_ids_trimmed = []
        for sentinel_id in sample_ids:
            sample_ids_trimmed.append(sentinel_id)
        
        # Convert the IDs to images
        scenes = []
        alert_date = None
        for i in range(0, len(sample_ids_trimmed)):
        
            # Get the id from the list of ids
            scene = ee.Image("COPERNICUS/S1_GRD/"+sample_ids_trimmed[i])
            
            # Append the scene to the list
            scenes.append(scene)
            
            # Get the alert date
            if i == 0:
                alert_date = scene.date()
            
        # Convert the list of images to an ee.ImageCollection and select the correct bands
        scenes = ee.ImageCollection.fromImages(scenes).select(self.output_bands)
               
        # Generate the features
        features = scenes.toBands().rename(self.model_feature_names)       
        
        # Load the GLAD Alert for the scene
        label = self.__retrieve_label(alert_date)
        
        # Stack the outputs
        labels_and_features = ee.Image.cat([features, label])
                
        # Run the sampling
        output = self.__sample_model_data(labels_and_features, sample.geometry())
        
        # Create the export filename
        file_name = 'alert_record_' + partition_id + '_' + str(sample_num)
        
        # Initiate the export 
        task = ee.batch.Export.table.toDrive(
            collection = ee.FeatureCollection([output]), 
            description = "Export-Mekong-Alert-" + str(sample_num), 
            folder = 'test_export', 
            fileNamePrefix = self.gcs_export_folder + '/' + model_set + '/' + file_name,
            fileFormat = 'CSV', 
            )

        # task = ee.batch.Export.table.toCloudStorage(
        #     collection = ee.FeatureCollection([output]),
        #     description = "Export-Mekong-Alert-" + str(sample_num),  
        #     bucket = self.gcs_bucket,
        #     fileNamePrefix = self.gcs_export_folder + '/' + model_set + '/' + file_name, 
        #     fileFormat = "TFRecord", 
        #     )
        
        # Check if the number of output features is correct
        # Number should be: self.model_feature_names + 2
        target_num_properties = len(self.model_feature_names) + 2 
        output_num_properties = output.propertyNames().length().getInfo()
        
        if target_num_properties == output_num_properties:
            print('Initating '+str(sample_num+1)+' of '+str(num_exports))
            
            print(output.get('angle_3').getInfo())
            
            # task.start()
        else:
            print('Export for index ' + str(sample_num+1) + ' had too few features.')

        # Log the info in the exporter
        self.export_monitor.add_task('export_'+str(sample_num), task)
    
        return None

if __name__ == "__main__":
    
    '''
    Ate -- only the parameters below need to be changed
    '''
    
    # The username associated with the Google Earth Engine account the script will be run on
    input_username = 'JohnBKilbride'
    
    # The Google Cloud Storage bucket to export the data too
    input_gcs_bucket = "kilbride_bucket_1"
    
    # The Name of the cloud storage folder to populate with the records
    input_gcs_export_folder = 'glad_alert_records'
    
    
    '''
    These can stay the same because I have shared the assets from the earlier stages of the script
    '''
    # A GEE FeatureCollection of sample points created by the script "0a_create_sample locations.py"
    input_sample_locations = ee.FeatureCollection("users/JohnBKilbride/SERVIR/real_time_monitoring/partition_points")
    
    # A GEE FeatureCollection containing the sample partitions created by the script "0a_create_sample locations.py"
    input_partitions = ee.FeatureCollection("users/JohnBKilbride/SERVIR/real_time_monitoring/partitions")
    
    # A projected coordinate system to use for the study area
    input_projection = ee.Projection('EPSG:32648')
    
    # The number of julian days to include after the most recent images' date
    input_forward_label_fuzz = 3
    
    # The number of julian days of GLAD alert records to include before a given images label date
    input_backward_label_fuzz = 10
    
    # The kernel Size
    input_kernel_size = 256

    # The number if sentinel-1 images to include in the exported feature tensor
    input_num_sentinel_images = 3
    
    # The Export ID to assign to the ee.FeatureCollection of GLAD feature groups to export 
    input_feat_group_export_id = "SERVIR/real_time_monitoring/glad_feature_groups"
    
    # The Export ID to assign to the ee.FeatureCollection of the GLAD labels to export
    input_glad_label_export_id = "SERVIR/real_time_monitoring/glad_labels"
    
    # The bands that will be included in the exported alert recoreds
    input_output_bands = ['VV','VH', 'angle']

    # Instantiate the object
    alert_generator = SyntheticAlertGenerator(input_sample_locations, input_partitions, input_username, input_projection, input_forward_label_fuzz, 
                                              input_backward_label_fuzz, input_kernel_size, input_gcs_bucket, input_gcs_export_folder, 
                                              input_num_sentinel_images, input_feat_group_export_id, input_glad_label_export_id,
                                              input_output_bands)
    
    # Aggregate the sentinel IDs needed for the second stage of processing
    alert_generator.aggregate_sar_for_alerts()
    
    
    
    # Generate the GLAD Labels
    # alert_generator.generate_glad_labels()
    
    # # Export the training dataset to google drive
    # start_time = datetime.now()
    # alert_generator.generate_synthetic_alert_dataset()
    # end_time = datetime.now()
    # print('')
    # print('Script time:', end_time - start_time)

    
    print('\nProgram completed.')

