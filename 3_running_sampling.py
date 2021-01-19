import ee
from time import sleep
from utils import task_monitor

ee.Initialize()

# Load the export monitor
EXPORT_MONITOR = task_monitor.GEETaskMonitor()

SHIFT_BEFORE = 15

def main ():
    
    # Define the username
    username = "JohnBKilbride"
    
    # Define the output location
    output_dir = "SERVIR/real_time_monitoring"
    
    # Define kernel size 
    kernel_size = 64
    image_kernel = get_kernel(kernel_size)
    
    # Cloud Storage Parameters
    cloud_bucket = "kilbride_bucket_1"
    cloud_folder = "glad_alert_records/"
    
    # Get the projection that is needed for the study area
    projection = ee.Projection('EPSG:32648')
    
    # Load in the GLAD Alert Images
    glad_alerts = ee.Image('users/JohnBKilbride/SERVIR/real_time_monitoring/glad_alerts_2019_to_2020')
    
    # Load in the sample locations
    sample_locations = ee.FeatureCollection("users/JohnBKilbride/SERVIR/real_time_monitoring/sample_locations_2019_2020_train_val_test_50k") \
        .randomColumn(None, 43214).sort('random').limit(5)
    
    #################################### Ignore stuff below #######################
    
    # Load in the topographic metrics
    topography = load_topographic()
    
    # Loop through the points
    num_samples = sample_locations.size().getInfo()
    sample_list = sample_locations.toList(num_samples)
    print('\nInitiating {num_exports} exports.'.format(num_exports=num_samples))
    for i in range(0, num_samples):
        
        print('\nExporting record {i}'.format(i=i))
        
        # Get the next feature by index and cast
        current_location = ee.Feature(sample_list.get(i))
        
        # Get the geometry from the current feature
        location_geo = current_location.geometry()
        
        # Get the dataset type
        dataset_subset = ee.Number(current_location.get('dataset_subset')).toInt8().getInfo()
        if dataset_subset == 1:
            dataset_name = 'train'
        elif dataset_subset == 2:
            dataset_name = 'validation'
        elif dataset_subset == 3:
            dataset_name = 'test'
        
        # Get the day and year of each of the dates
        before_day = ee.Number(current_location.get('start_day'))  
        before_year = ee.Number(current_location.get('start_year'))
        after_day = ee.Number(current_location.get('alert_day'))
        after_year = ee.Number(current_location.get('alert_year'))
        
        # Get the Before and After Images
        sentinel = generate_before_after_image(before_day, after_day, before_year, after_year, location_geo)
        
        # Get the Labels
        glad_label = calculate_glad_label(glad_alerts, before_day, after_day, before_year, after_year)
        
        # Combine the labels
        all_bands = ee.Image.cat(sentinel, topography, glad_label).toFloat()
        
        # # Sample with point and neighborhoodToArray()
        # arrays = all_bands.neighborhoodToArray(image_kernel)
        # output = arrays.sampleRegions(
        #     collection = ee.FeatureCollection([current_location]),
        #     properties = ["VV_before","VH_before","VV_after","VH_after","glad_alert"],
        #     scale = 10, 
        #     projection = projection, 
        #     geometries = False
        #     )
            
        # # Export the example to Google Cloud Storage
        # task_1 = ee.batch.Export.table.toCloudStorage(
        #     collection = output, 
        #     description = 'Export-GLAD-' + str(i), 
        #     bucket = cloud_bucket, 
        #     fileNamePrefix = cloud_folder + '/glad_alert_' + str(i), 
        #     fileFormat = 'TFRecord', 
        #     selectors = ["VV_before","VH_before","VV_after","VH_after","glad_alert"]
        #     )
        # task_1.start()
        
        # For debugging
        # if i < 10:
        export_geometry = location_geo.buffer(ee.Number(kernel_size/2).multiply(10), ee.ErrorMargin(0.0001, "projected"), projection) \
                .bounds(ee.ErrorMargin(0.0001, "projected"), projection)
        task_2 = ee.batch.Export.image.toDrive(
            image = all_bands, 
            description = 'GLAD-Image-Export-' + str(i),
            folder = 'alert_' + str(kernel_size) + '_' + str(kernel_size) + '_before_after_delta_topo',
            fileNamePrefix = 'glad_'+ dataset_name + '_' + str(i), 
            region = export_geometry, 
            scale = 10,
            crs = projection,
            maxPixels = 1e6
            )
        task_2.start()
        
        # Check for completed exports every 250 iterations
        if  i % 250 == 0:
            check_for_monitor_capacity()
            
        EXPORT_MONITOR.add_task('export_' + str(i), task_2)
            
    # Run the monitoring of the exports
    EXPORT_MONITOR.monitor_tasks()
    EXPORT_MONITOR.reset_monitor()
    print('...export completed.')    
            
    return None

# Produces a kernel of a given sized fro sampling in GEE
def get_kernel (kernel_size):
    eelist = ee.List.repeat(1, kernel_size)
    lists = ee.List.repeat(eelist, kernel_size)
    kernel = ee.Kernel.fixed(kernel_size, kernel_size, lists)
    return kernel

# Scale the integer values to a range between 1 and 0
def scale_sentinel_values (image):
    return image.unmask(-50).clamp(-50, 1).unitScale(-50, 1).set('system:time_start', image.date())

# Generates topographic metrics (globally) with the SRTM dataset
def load_topographic ():
  
    # Load in the SRTM dem
    srtm = ee.Image("USGS/SRTMGL1_003")
    
    # Topographic methods
    metrics = ee.Algorithms.Terrain(srtm).select(['elevation', 'slope', 'aspect'])
    
    return metrics.toFloat()

# Write a script that will do the following
# Take a parameter for the start date and the end date of a composite
def generate_before_after_image (start_alert_date, end_alert_date, start_alert_year, end_alert_year, alert_position):
    
    # Compute a buffer around the alert area
    alert_area = alert_position.buffer(250).bounds()
    
    # Load the Sentinel 1 GRD Image collectiona nd apply that 
    # fancy ass transformation
    sentinel_1 = ee.ImageCollection("COPERNICUS/S1_GRD") \
        .filterBounds(alert_area) \
        .select(['VV','VH']) \
        .map(scale_sentinel_values)
    
    # Get the composite of the area and after the alert
    before = create_before_image(sentinel_1, start_alert_year, start_alert_date, alert_position)
    after = create_after_image(sentinel_1, end_alert_year, end_alert_date, alert_position)
    before = ee.Image(before).toFloat()
    after = ee.Image(after).toFloat()
    
    # Compute the difference of the two bands
    delta = after.select(['VV_after','VH_after'], ['VV','VH']) \
        .subtract(before.select(['VV_before','VH_before'], ['VV','VH'])) \
        .rename(['VV_delta', 'VH_delta']) \
        .toFloat()
    
    # Get the composite of the area after the alert
    sentinel = ee.Image.cat(before, after, delta) \
        .set('system:time_start', ee.Image(before).date().millis()) \
        .set('system:time_end', ee.Image(after).date().millis()) \
        .toFloat()
    
    return sentinel
    
# Create an image to represent the pre-disturbance period
def create_before_image (sentinel, year, alert_date, geometry):
    
    # Filter the collection to include sentinel 1 images up to the 
    # end of the pre-disturbane period
    end_date = ee.Date.fromYMD(year, 1, 1).advance(ee.Number(alert_date), 'day')
    filtered = sentinel.filterDate('2018-01-01', end_date) \
        .sort('system:time_start', False) \
        .reduce(ee.Reducer.firstNonNull()) \
        .select(['VV_first','VH_first'], ['VV_before','VH_before'])
    
    return filtered.set('system:time_start', end_date.millis())

# Create an image to represent the post-disturbance period
def create_after_image (sentinel, year, alert_date, geometry):

    # Filter the collection to include sentinel 1 images that occured
    # after the disturbance event
    start_date = ee.Date.fromYMD(year, 1, 1).advance(ee.Number(alert_date), 'day')
    filtered = sentinel.filterDate(start_date, '2022-01-01') \
        .sort('system:time_start') \
        .reduce(ee.Reducer.firstNonNull()) \
        .select(['VV_first','VH_first'], ['VV_after','VH_after'])
        
    return filtered.set('system:time_start', start_date.millis())

# Calculate the GLAD label for the GLAD Alerts
def  calculate_glad_label(glad_alerts, before_day, after_day, before_year, after_year):
    
    # # Increase the before date by an arbitrary constant
    before_year = before_year.add(SHIFT_BEFORE)
    
    # Process the 2019 label
    image_2019_a = glad_alerts.select('alertDate19').gte(before_day).And(ee.Number(before_year).eq(2018).Or(ee.Number(before_year).eq(2019)))
    image_2019_b = glad_alerts.select('alertDate19').lte(after_day).And(ee.Number(after_year).lte(2019))
    image_2019 = image_2019_a.And(image_2019_b)
    
    # Process the 2019 label
    image_2020_a = glad_alerts.select('alertDate20').gte(before_day).And(ee.Number(before_year).eq(2020))
    image_2020_b = glad_alerts.select('alertDate20').lte(after_day).And(ee.Number(after_year).lte(2020))
    image_2020 = image_2020_a.And(image_2020_b)
    
    # Create the start and end time as ee.Date objects
    start_date = ee.Date.fromYMD(ee.Number(before_year), 1, 1).advance(ee.Number(before_day), 'day')
    end_date = ee.Date.fromYMD(ee.Number(after_year), 1, 1).advance(ee.Number(after_day), 'day')
    
    # Combine the two labels
    combined = image_2019.add(image_2020) \
        .gte(1).rename(['glad_alert']).set({
        'system:time_start': start_date,
        'system:time_end': end_date
        }).toByte()
    
    return combined

def check_for_monitor_capacity():

    # Compute the current capacity of the monitor
    capacity = EXPORT_MONITOR.get_monitor_capacity()
        
    # If monitor is less than 5% away from its maximum capacity then wait.
    if capacity > 0.95:
        while capacity > 0.95:
            print("...Monitor capacity at " + str(round(capacity * 100, 3)))
            sleep(600)
            EXPORT_MONITOR.check_status()
            capacity = EXPORT_MONITOR.get_monitor_capacity()
            
    return None

if __name__ == "__main__":
    print('Program started..')
    main()
    print('\nProgram completed.')



