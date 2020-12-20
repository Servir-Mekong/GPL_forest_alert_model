import ee

ee.Initialize()

def generate_sentinel_before_after (start_alert_date, end_alert_date, start_alert_year, end_alert_year, alert_position):
    '''
    Write a script that will do the following
    Take a parameter for the start date and the end date of a composite
    '''

    # Compute a buffer around the alert area
    alert_area = alert_position.buffer(250).bounds()
    
    # Load the Sentinel 1 GRD Image collectiona nd apply that 
    # fancy ass transformation
    sentinel_1 = ee.ImageCollection("COPERNICUS/S1_GRD") \
        .filterBounds(alert_area) \
        .select(['VV','VH']) \
        .map(to_natural)
    
    # Get the composite of the area and after the alert
    before = create_before_image(sentinel_1, start_alert_year, start_alert_date, alert_position)
    after = create_after_image(sentinel_1, end_alert_year, end_alert_date, alert_position)
    before = ee.Image(before).toFloat()
    after = ee.Image(after).toFloat()
    
    # Get the composite of the area after the alert
    sentinel = ee.Image.cat(before, after) \
        .set('system:time_start', ee.Image(before).date().millis()) \
        .set('system:time_end', ee.Image(after).date().millis()) \
        .toFloat()
    
    return sentinel
    
def to_natural (img):
    '''def to convert from dB to natural.'''
    return ee.Image(10.0).pow(img.divide(10.0)).set('system:time_start', img.date().millis())

def create_before_image (sentinel, year, alert_date, geometry):
    '''Create an image to represent the pre-disturbance period.'''
    # Filter the collection to include sentinel 1 images up to the 
    # end of the pre-disturbane period
    end_date = ee.Date.fromYMD(year, 1, 1).advance(ee.Number(alert_date), 'day')
    filtered = sentinel.filterDate('2018-01-01', end_date) \
        .sort('system:time_start') \
        .reduce(ee.Reducer.firstNonNull()) \
        .select(['VV_first','VH_first'], ['VV_before','VH_before'])
    
    return filtered.set('system:time_start', end_date.millis())

def create_after_image (sentinel, year, alert_date, geometry):
    '''Create an image to represent the post-disturbance period.'''
    # Filter the collection to include sentinel 1 images that occured
    # after the disturbance event
    start_date = ee.Date.fromYMD(year, 1, 1).advance(ee.Number(alert_date), 'day')
    filtered = sentinel.filterDate(start_date, '2022-01-01') \
        .sort('system:time_start', False) \
        .reduce(ee.Reducer.firstNonNull()) \
        .select(['VV_first','VH_first'], ['VV_after','VH_after'])
        
    return filtered.set('system:time_start', start_date.millis())

if __name__ == "__main__":
    
    input_start_alert_date = 167
    input_end_alert_date = 284
    input_start_alert_year = 2020
    input_end_alert_year = 2020
    input_alert_position = ee.Geometry.Point([105.62835785245568, 13.140949140783269]);
    
    output = generate_sentinel_before_after(input_start_alert_date, input_end_alert_date, input_start_alert_year, input_end_alert_year, input_alert_position)
    print(output.getInfo())
    