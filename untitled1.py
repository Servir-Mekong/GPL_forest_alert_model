import ee

ee.Initialize()
 
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

class ExportAlertDataset


def main ():

    # Define the projection, the export location (google drive)
    forward_label_fuzz = 7
    backward_label_fuzz = 7
    kernel_size = 256
    export_folder = 'SERVIR_alert_data'
    
    # Things that will become parametesr later
    sample_groups = ee.FeatureCollection("users/JohnBKilbride/SERVIR/real_time_monitoring/feature_groups") \
        .map(id_list_to_string) \
        .limit(10)
    num_features = sample_groups.size().getInfo()
    print('Num exports:', num_features)
    
    # Get the bounds of the area with 
    study_area = sample_groups.geometry().bounds()
    
    # Load the GLAD Alerts
    glad_alerts = load_formatted_alerts(study_area)
    
    # Load the Sentinel 1 GRD dataset
    sentinel = ee.ImageCollection("COPERNICUS/S1_GRD") \
        .filterBounds(study_area) \
        .filterMetadata('orbitProperties_pass', 'equals', 'DESCENDING') \
        .filterDate('2018-01-01', '2019-12-31') \
        .select(['VV','VH'])
    
    # Load the kernel
    kernel = create_kernel(kernel_size)
    
    # Load the features
    feature_names = ["VV_1","VH_1","VV_2","VH_2","VV_3","VH_3"]
    
    # Loop over the features
    sample_group_list = sample_groups.toList(1e7)
    for i in range(0, num_features):
    
        # Get the feature
        feature = ee.Feature(sample_group_list.get(i))
        
        # Run the sampling and export the feature export
        export_dataset_sample(feature, i, sentinel, glad_alerts, forward_label_fuzz, backward_label_fuzz, kernel, feature_names, export_folder)
        
    return None


# Convert the string list into a list
def id_list_to_string (feature):

    # Cast the feature
    feature = ee.Feature(feature)
    
    # Get the list of strings and disucss
    id_list = ee.String(feature.get('ordered_sentinel_ids')).replace("\\[","","g").replace("\\]","","g").split(',')
    
    return feature.set('id_list', id_list)

# Generate the kernel
def create_kernel (kernel_size):
    kernel_cols = ee.List.repeat(1, kernel_size)
    kernel_matrix = ee.List.repeat(kernel_cols, kernel_size)
    kernel = ee.Kernel.fixed(kernel_size, kernel_size, kernel_matrix)
    return kernel

# Main logic of the script
def load_formatted_alerts (study_area):

    # Load in the GLAD forest alerts
    # Information: http:#glad-forest-alert.appspot.com/
    glad_alerts = ee.ImageCollection('projects/glad/alert/2019final').filterBounds(study_area)
    
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


#  Turn the single-image GLAD alert dates intoa  time-series of 365 binary masks
def glad_alert_to_collection (glad_alert, year, alert_band_name):
    
    # Create a list of dates
    days = ee.List.sequence(1, 365)
    
    # Map over the days to create the alert time-series
    def inner_map (day):
    
        # Cast the day as an ee.Number
        day = ee.Number(day).toInt16()
        
        # Create the date stamp
        img_date = ee.Date.fromYMD(year, 1, 1).advance(day, 'day').millis()
        
        # Get where the alerts are the 
        julian_alert = glad_alert.select(alert_band_name).eq(day).set('system:time_start', img_date) \
            .rename(['glad_alert_binary'])
        
        return julian_alert.toByte()


    return ee.ImageCollection.fromImages(days.map(inner_map))

#  Create a series of all zero binary masks
def create_dummy_alerts (year):

    # Create a list of dates
    days = ee.List.sequence(1, 365)
    
    # Map over the days to create the alert time-series
    def inner_map (day):
    
        # Cast the day as an ee.Number
        day = ee.Number(day).toInt16()
        
        # Create the date stamp
        img_date = ee.Date.fromYMD(year, 1, 1).advance(day, 'day').millis()
        
        # Get where the alerts are the 
        julian_alert = ee.Image(0).set('system:time_start', img_date) \
            .rename(['glad_alert_binary'])
        
        return julian_alert.toByte()
    
    return ee.ImageCollection.fromImages(days.map(inner_map))

# Convert GLAD alert into a binary
def glad_to_label (glad_alerts, alert_date, fuzz_forward, fuzz_backward):

    # Fuzz the start and the end_date
    start_date_fuzz = alert_date.advance(-1 * fuzz_backward, 'day')
    end_date_fuzz = alert_date.advance(fuzz_forward, 'day')
    
    # Create the label from the glad_inputs and the fuzzed dates
    label = glad_alerts.filterDate(start_date_fuzz, end_date_fuzz).max().rename(['glad_alert'])
    
    return label

# Sample the model data
def sample_model_data (stack, kernel, sample_point):

    # Select the points with the matching ID
    sample_points = ee.FeatureCollection([sample_point])
    
    # Convert the stack to an array neighborhood
    stack_array = stack.neighborhoodToArray(
        kernel = kernel, 
        defaultValue = 0
        )
    
    # Run the sampling proceedure
    samples = stack_array.reduceRegions(
        collection = sample_points, 
        reducer = ee.Reducer.first(), 
        scale = 25, 
        tileScale = 1
        )
    
    return ee.Feature(samples.first())


# Function exports an individual sample to goolgle drive as a TFRecord,. Thesse can be combined as a TensorFlow Dataset
def export_dataset_sample (sample, sample_num, sentinel_images, glad_alerts, fuzz_forward, fuzz_backward, kernel, feature_names,
export_folder_name):

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
        scenes.append(scene)
    
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
        fileNamePrefix =  file_name + str(sample_num), 
        fileFormat =  "TFRecord"
        )
    
    task.start()
    
    # Log the info in the exporter 
    
    
    return None


main()



