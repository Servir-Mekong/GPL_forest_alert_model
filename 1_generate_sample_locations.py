import ee
from time import sleep

ee.Initialize()

def main ():

    print('sleeping...')
    sleep(60 * 60 * 2)
    
    # Load in the pre-processed GLAD alerts
    glad_alerts = ee.Image('users/JohnBKilbride/SERVIR/real_time_monitoring/glad_alerts_2019_to_2020')
        
    # Get the projection that is needed for the study area
    projection = ee.Projection('EPSG:32648')
    
    # Define the username
    username = "JohnBKilbride"
    
    # Define the output location
    output_dir = "SERVIR/real_time_monitoring"
    
    # Kernel size (# of pixels)
    kernel_size = 64
    
    # Compute the kernel radius
    kernel_radius = ee.Number(kernel_size).divide(2)
    
    # Get the study area
    study_area = ee.Geometry.Polygon([[[104.0311, 14.3134],[104.0311, 12.5128],[106.0416, 12.5128],[106.0416, 14.3134]]], None, False)
    
    # Seperate the 2019 and 2020 glad data
    glad_2019 = glad_alerts.select(['alertBinary19', 'alertDate19']) \
        .addBands(ee.Image.constant(2019).rename('year')) \
        .select(["alertBinary19","alertDate19", "year"],["binary","alert_day", "alert_year"]) \
        .toInt16()
    glad_2020 = glad_alerts.select(['alertBinary20', 'alertDate20']) \
        .addBands(ee.Image.constant(2020).rename('year')) \
        .select(["alertBinary20","alertDate20", "year"],["binary","alert_day", "alert_year"]) \
        .toInt16()
    
    # Take a stratified random sample of the 2019 layer 
    sample_2019 = get_sample_of_disturbances(glad_2019, projection, study_area)
    sample_2020 = get_sample_of_disturbances(glad_2020, projection, study_area)
    
    # Merge the two different samples
    combined_samples = sample_2019.merge(sample_2020)
    
    # Add the "start date" to each of the images
    # This represents the first pre-disturbance observation that was actually valid (uses Landsat QA bands)
    output = ee.FeatureCollection(add_start_date(combined_samples)) \
        .select(['alert_day','alert_year','start_day','start_year']) 
        
    # Apply a random displacement to each of the point locations 
    output = apply_displacement(output, projection, kernel_radius)
        
    # Export the sample locations with the julian date of the disturbance to google drive
    task = ee.batch.Export.table.toAsset(
        collection = output, 
        description = "Sample-Points-GLAD", 
        assetId = "users/"+username+"/"+output_dir+"/sample_locations_2019_2020_50k"
        )
    task.start()
    
    return None

def add_start_date (sample_points):
    '''Get the timing for the "before" image'''  
    # Load in the landsat imagery
    landsat = load_landsat_imagery(sample_points.geometry().bounds())
    
    # Mapped function to apply over the sample_points ee.FeatureCollection
    def inner_map (sample_point):
    
        # Cast the input
        sample_point = ee.Feature(sample_point)
        
        # Get the GLAD alert day and year
        alert_day = ee.Number(sample_point.get("alert_day"))
        alert_year = ee.Number(sample_point.get("alert_year"))
        
        # Construct the alert date as an ee.Date object
        glad_alert_date = ee.Date.fromYMD(ee.Number(alert_year), 1, 1) \
            .advance(ee.Number(alert_day).subtract(1), 'day')
        
        # Get the start date
        start_day = ee.Number(glad_alert_date.getRelative('day', 'year'))
        start_year = ee.Number(glad_alert_date.get('year'))
        
        # Append the sampled values to the original feature
        output = sample_point.set({
            "start_day": start_day,
            "start_year": start_year
            })
        
        return output
        
    return sample_points.map(inner_map)

def load_landsat_imagery(input_geometry):
    '''Loads the total coverage for the study area for the Landsat SR T1 product'''
    # Load in the Landsat 7 Imagery
    landsat_7 = ee.ImageCollection("LANDSAT/LE07/C01/T1_SR") \
        .filterBounds(input_geometry) \
        .filterDate('2018-01-01', '2022-12-31') \
        .select(['pixel_qa']) \
        .map(qa_to_date)
    
    # Load in teh Landsat 8 imagery
    landsat_8 = ee.ImageCollection("LANDSAT/LC08/C01/T1_SR") \
        .filterBounds(input_geometry) \
        .filterDate('2018-01-01', '2022-12-31') \
        .select(['pixel_qa']) \
        .map(qa_to_date)
    
    # Merge the two collections
    combined = landsat_7.merge(landsat_8)
    
    return ee.ImageCollection(combined)

def qa_to_date (image):
    '''Function applies the Landsat SR cloud mask (CFMask output).'''
    # Get the date from the image
    date = image.date()
    
    # Convert the QA band into a binary layer
    qa = image.select('pixel_qa')
    mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 5).eq(0))
    
    # Create the julian date band
    day = ee.Image.constant(date.getRelative('day', 'year')).rename('qa_day')
    year = ee.Image.constant(date.get('year')).rename('qa_year')
    
    return day.addBands(year).updateMask(mask) \
        .select(['qa_day', 'qa_year']) \
        .set('system:time_start', date.millis()) \
        .toInt16()
        
def apply_displacement(features, projection, kernel_radius):

    # Get the original band names for later
    orig_prop_names = features.first().propertyNames()
    
    # Add the two random column
    features_random = features.randomColumn('random_x').randomColumn('random_y')
    
    # Apply an inner function which 
    def inner_map (point):
    
        # Cast the point
        point = ee.Feature(point)
        
        # Get the geometry from the point
        point_geo = point.geometry()
        
        # Get the displacement amounts
        x_translate = ee.Number(point.get('random_x')).subtract(0.5).multiply(10).multiply(ee.Number(kernel_radius))
        y_translate = ee.Number(point.get('random_y')).subtract(0.5).multiply(10).multiply(ee.Number(kernel_radius))
        
        # Apply the displacement to the projection
        prj_trans = projection.translate(x_translate, y_translate)
        new_point_geo = ee.Geometry.Point(point_geo.transform(prj_trans).coordinates(), projection)
        
        return point.setGeometry(new_point_geo)
    
    return features_random.map(inner_map).select(orig_prop_names)

# Gets a random sample of disturbance locations. this funmction returns an 
# ee.FeatureCollection where each point retains its geometry
def get_sample_of_disturbances (image, projection, study_area):
    
    # Get a sample of disturbance locations
    samples = image.stratifiedSample(
        numPoints = 25000, 
        classBand = 'binary', 
        region = study_area, 
        scale = 10, 
        projection = projection, 
        seed = 57992, 
        classValues = [0, 1], 
        classPoints = [0, 25000],
        dropNulls = True, 
        tileScale = 1, 
        geometries = True
        )
    
    return samples


if __name__ == "__main__":
    print("Beginning script...")
    main()
    print("\nProgram completed.")
    
    
    