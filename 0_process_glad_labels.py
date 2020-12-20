import ee

ee.Initialize()

def main ():
    
    # Define the study area
    study_area = ee.Geometry.Polygon([[[101.4915, 15.4685],[101.4915, 9.3992],[108.8084, 9.3992],[108.8084, 15.468]]], None, False)
    
    # Define the username
    username = "JohnBKilbride"
    
    # Define the projection of the study area
    projection = ee.Projection('EPSG:32648')
    
    # Define the output location
    output_dir = "SERVIR/real_time_monitoring"
    
    # Generate the outputs
    labels = generate_glad_label(study_area, projection)
        
    # Export the data as an asset
    task = ee.batch.Export.image.toAsset(
        image = labels.toInt16(), 
        description = "Processed-Glad-Alerts", 
        assetId ='users/'+username+"/"+output_dir+'/glad_alerts_2019_to_2020', 
        region = study_area.bounds(), 
        scale = 30, 
        crs = projection,  
        maxPixels = 1e13
        )
    task.start()
        
    return None

# Apply the minimum mapping unit to the images
def compute_mmu_mask (image, band, projection):

    # Compute the connected pixels layer the 
    output = image.select([band]) \
        .connectedPixelCount(8, True) \
        .gt(4).reproject(projection, None, 30) \
        .toInt8()
    
    return output

def generate_glad_label (study_area, projection):

    # Load in the glad alerts from the different sources
    glad_alerts_2019 = ee.ImageCollection('projects/glad/alert/2019final') \
        .filterBounds(study_area)
    glad_alerts_2020 = ee.ImageCollection('projects/glad/alert/UpdResult') \
        .filterBounds(study_area)
    
    # Isolate the 2019 and 2020 alerts
    alerts_2019 = ee.Image(glad_alerts_2019.select(['alertDate19']) \
        .sort('system:time_start', False) \
        .first()).toInt16()
    alerts_2020 = ee.Image(glad_alerts_2020.select(['alertDate20']) \
        .sort('system:time_start', False) \
        .first()).toInt16()               
    
    # Create the set of binary "alert/no alert images"
    binary_2019 = alerts_2019.neq(0).rename(['alertBinary19']).multiply(2019).toInt16()
    binary_2020 = alerts_2020.neq(0).rename(['alertBinary20']).multiply(2020).toInt16()
    
    # Apply the minimum mapping unit
    mmu_mask_2019 = compute_mmu_mask(binary_2019, 'alertBinary19', projection)
    mmu_mask_2020 = compute_mmu_mask(binary_2020, 'alertBinary20', projection)
    
    # Apply the minimum mapping unit mask
    alerts_2019 = alerts_2019.reproject(projection, None, 30).where(mmu_mask_2019.Not(), 0)
    alerts_2020 = alerts_2020.reproject(projection, None, 30).where(mmu_mask_2020.Not(), 0)
    binary_2019 = binary_2019.reproject(projection, None, 30).where(mmu_mask_2019.Not(), 0)
    binary_2020 = binary_2020.reproject(projection, None, 30).where(mmu_mask_2020.Not(), 0)
    
    # Combine all of the different bands
    combined = ee.Image.cat(alerts_2019, binary_2019, alerts_2020, binary_2020)
    
    return combined

if __name__ == "__main__":

    print('Starting program...')
    main()
    print('\nProgram completed.')

