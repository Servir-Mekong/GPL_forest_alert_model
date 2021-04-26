import ee
from time import sleep
from utils import task_monitor
import math
import numpy as np
import random
from numpy.random import seed
from numpy.random import rand
import sys

ee.Initialize()

seed(10)
values = rand(50000)

# Load the export monitor
EXPORT_MONITOR = task_monitor.GEETaskMonitor()

def main ():
    
    # set some ee.Model parameters
    featureNames = ['VH_after0','VH_after1','VH_before0', 'VH_before1','VH_before2','VV_after0','VV_after1', 'VV_before0', 'VV_before1', 'VV_before2', 'glad_alert', 'non_alert']
    bands = ['VH_after0', 'VH_before0', 'VH_before1', 'VH_before2', 'VV_after0','VV_before0', 'VV_before1', 'VV_before2']
    PROJECT = 'projectname';
    MODEL_NAME = 'alerts';
    VERSION_NAME = 'modelName';

    # Load the trained model and use it for prediction.
    model = ee.Model.fromAiPlatformPredictor(
    projectName= PROJECT,
    modelName= MODEL_NAME,
    version= VERSION_NAME,
    inputTileSize= [128,128],
    inputOverlapSize= [16,16],
    proj= ee.Projection('EPSG:4326').atScale(10),
    fixInputProj= True,
    outputBands= {'landclass': {'type': ee.PixelType.float(),'dimensions': 1 }})
    MODE = 'DESCENDING'

    protected1 = ee.FeatureCollection("projects/cemis-camp/assets/wdpa/WDPA_WDOECM_KHM_shp-polygons").filter(ee.Filter.eq("NAME","Prey Lang"))
    protected2 = ee.FeatureCollection("projects/cemis-camp/assets/wdpa/WDPA_WDOECM_KHM_shp-polygons0").filter(ee.Filter.eq("NAME","Boeng Paer"))

    geometry = protected1.merge(protected2).geometry()

    # Get the projection that is needed for the study area
    projection = ee.Projection('EPSG:32648')
    
    year = 2016
    startDate = ee.Date.fromYMD(year,1,1)
    endDate = ee.Date.fromYMD(year,6,1)
    
    # Import Sentinel-1 Collection 
    s1 =  ee.ImageCollection('COPERNICUS/S1_GRD')\
			.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
			.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
			.filter(ee.Filter.eq('orbitProperties_pass', MODE))\
			.filter(ee.Filter.eq('instrumentMode', 'IW'))\
			.filterBounds(geometry)\
			.map(erodeGeometry)\
			.map(terrainCorrection)\
			.map(applySpeckleFilter)\
			.map(addRatio)
			


    s1After = s1.filterDate(startDate,endDate)
    s = s1After.size().getInfo()

    s1After = s1After.toList(500)

    for i in range(0,int(s),1):
	afterImg = ee.Image(s1After.get(i))
	geom = afterImg.geometry()
	prop = afterImg.toDictionary()

	name = afterImg.get("system:index").getInfo()
	date = ee.Date(afterImg.get("system:time_start")) 


	s1Before = s1.filterBounds(geom)
	before = createSeriesBefore(s1Before,date.advance(-1,"days"))
	after = afterImg.select(["VV","VH"],["VV_after0","VH_after0"])

	image = before.addBands(after).unmask(0,False)
	image = image.select(bands).toFloat()

	prediction = ee.Image(model.predictImage(image.toArray()).arrayFlatten([["landclass","other"]]).toFloat())
	prediction = prediction.select("landclass").multiply(100).toInt()   
    
	outputName = "projects/cemis-camp/assets/GPLforestAlerts/" + name
	
	geom = geom.transform("EPSG:4326",0.01)
	output = ee.Image(prediction).clip(geom)
	output = clipEdge(output)
	output = output.set(prop)
	output = output.set("system:time_start",date)


	geom = geom.getInfo()
	task_ordered = ee.batch.Export.image.toAsset(image=output, description="alerts "+ name, assetId=outputName,region=geom['coordinates'], maxPixels=1e13,scale=10 )
	task_ordered.start()



def createSeriesBefore(collection,date,iters=3,nday =12):

    iterations = range(1,iters*nday,nday)
    names = ["_before{:01d}".format(x) for x in range(0,iters,1)]
    print(iterations)
    def returnCollection(day,name):
	start = ee.Date(date).advance(-day,"days").advance(-nday,"days")
	end = ee.Date(date).advance(-day,"days")
	bandNames = ["VV"+name,"VH"+name]
	return ee.Image(collection.filterDate(start,end).mean())\
				  .select(["VV","VH"],bandNames)\
				  .set("system:time_start",start)
    
    return toBands(ee.ImageCollection.fromImages(map(returnCollection,iterations,names)))

def createSeriesAfter(collection,date,iters=2,nday =12):
    
    iterations = range(1,iters*nday,nday)
    names = ["_after{:01d}".format(x) for x in range(0,iters,1)]
  
    def returnCollection(day,name):
	start = ee.Date(date).advance(day,"days")
	end = ee.Date(date).advance(day+nday,"days")
	bandNames = ["VV"+name,"VH"+name]
	return ee.Image(collection.filterDate(start,end).mean())\
				  .select(["VV","VH"],bandNames)\
				  .set("system:time_start",start)
    
    return toBands(ee.ImageCollection.fromImages(map(returnCollection,iterations,names)))    


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

# Implementation by Andreas Vollrath (ESA), inspired by Johannes Reiche (Wageningen)
def terrainCorrection(image):
    date = ee.Date(image.get('system:time_start'))
    imgGeom = image.geometry()
    srtm = ee.Image('USGS/SRTMGL1_003').clip(imgGeom)  # 30m srtm 
    sigma0Pow = ee.Image.constant(10).pow(image.divide(10.0))

    #Article ( numbers relate to chapters) 
    #2.1.1 Radar geometry 
    theta_i = image.select('angle')
    phi_i = ee.Terrain.aspect(theta_i).reduceRegion(ee.Reducer.mean(), theta_i.get('system:footprint'), 1000).get('aspect')

    #2.1.2 Terrain geometry
    alpha_s = ee.Terrain.slope(srtm).select('slope')
    phi_s = ee.Terrain.aspect(srtm).select('aspect')

    # 2.1.3 Model geometry
    # reduce to 3 angle
    phi_r = ee.Image.constant(phi_i).subtract(phi_s)

    #convert all to radians
    phi_rRad = phi_r.multiply(math.pi / 180)
    alpha_sRad = alpha_s.multiply(math.pi / 180)
    theta_iRad = theta_i.multiply(math.pi / 180)
    ninetyRad = ee.Image.constant(90).multiply(math.pi / 180)

    # slope steepness in range (eq. 2)
    alpha_r = (alpha_sRad.tan().multiply(phi_rRad.cos())).atan()

    # slope steepness in azimuth (eq 3)
    alpha_az = (alpha_sRad.tan().multiply(phi_rRad.sin())).atan()

    # local incidence angle (eq. 4)
    theta_lia = (alpha_az.cos().multiply((theta_iRad.subtract(alpha_r)).cos())).acos()
    theta_liaDeg = theta_lia.multiply(180 / math.pi)
  
    # 2.2 
    # Gamma_nought_flat
    gamma0 = sigma0Pow.divide(theta_iRad.cos())
    gamma0dB = ee.Image.constant(10).multiply(gamma0.log10())
    ratio_1 = gamma0dB.select('VV').subtract(gamma0dB.select('VH'))

    # Volumetric Model
    nominator = (ninetyRad.subtract(theta_iRad).add(alpha_r)).tan()
    denominator = (ninetyRad.subtract(theta_iRad)).tan()
    volModel = (nominator.divide(denominator)).abs()

    # apply model
    gamma0_Volume = gamma0.divide(volModel)
    gamma0_VolumeDB = ee.Image.constant(10).multiply(gamma0_Volume.log10())

    # we add a layover/shadow maskto the original implmentation
    # layover, where slope > radar viewing angle 
    alpha_rDeg = alpha_r.multiply(180 / math.pi)
    layover = alpha_rDeg.lt(theta_i);

    # shadow where LIA > 90
    shadow = theta_liaDeg.lt(85)

    # calculate the ratio for RGB vis
    ratio = gamma0_VolumeDB.select('VV').subtract(gamma0_VolumeDB.select('VH'))

    output = gamma0_VolumeDB.addBands(ratio).addBands(alpha_r).addBands(phi_s).addBands(theta_iRad)\
			    .addBands(layover).addBands(shadow).addBands(gamma0dB).addBands(ratio_1)

    return output.select(['VV', 'VH'], ['VV', 'VH']).set("system:time_start",date).clip(imgGeom ).copyProperties(image)


#
# * Clips 5km edges
# */
def erodeGeometry(image):
    return image.clip(image.geometry().buffer(-5000))

def clipEdge(image):
    return image.clip(image.geometry().buffer(-1000))


def applySpeckleFilter(img):
    
    vv = img.select('VV')
    vh = img.select('VH')
    vv = speckleFilter(vv).rename('VV');
    vh = speckleFilter(vh).rename('VH');
    return ee.Image(ee.Image.cat(vv,vh).copyProperties(img,['system:time_start'])).clip(img.geometry()).copyProperties(img);


def speckleFilter(image):
    """ apply the speckle filter """
    ksize = 3
    enl = 7; 
    
    geom = image.geometry()
    
    # Convert image from dB to natural values
    nat_img = toNatural(image);

    # Square kernel, ksize should be odd (typically 3, 5 or 7)
    weights = ee.List.repeat(ee.List.repeat(1,ksize),ksize);

    # ~~(ksize/2) does integer division in JavaScript
    kernel = ee.Kernel.fixed(ksize,ksize, weights, ~~(ksize/2), ~~(ksize/2), False);

    # Get mean and variance
    mean = nat_img.reduceNeighborhood(ee.Reducer.mean(), kernel);
    variance = nat_img.reduceNeighborhood(ee.Reducer.variance(), kernel);

    # "Pure speckle" threshold
    ci = variance.sqrt().divide(mean);# square root of inverse of enl

    # If ci <= cu, the kernel lies in a "pure speckle" area -> return simple mean
    cu = 1.0/math.sqrt(enl);

    # If cu < ci < cmax the kernel lies in the low textured speckle area
    # -> return the filtered value
    cmax = math.sqrt(2.0) * cu;

    alpha = ee.Image(1.0 + cu*cu).divide(ci.multiply(ci).subtract(cu*cu));
    b = alpha.subtract(enl + 1.0);
    d = mean.multiply(mean).multiply(b).multiply(b).add(alpha.multiply(mean).multiply(nat_img).multiply(4.0*enl));
    f = b.multiply(mean).add(d.sqrt()).divide(alpha.multiply(2.0));

    # If ci > cmax do not filter at all (i.e. we don't do anything, other then masking)
    # Compose a 3 band image with the mean filtered "pure speckle", 
    # the "low textured" filtered and the unfiltered portions
    out = ee.Image.cat(toDB(mean.updateMask(ci.lte(cu))),toDB(f.updateMask(ci.gt(cu)).updateMask(ci.lt(cmax))),image.updateMask(ci.gte(cmax)));	
		
    return out.reduce(ee.Reducer.sum()).clip(geom);

def addRatio(img):
    geom = img.geometry()
    vv = toNatural(img.select(['VV'])).rename(['VV']);
    vh = toNatural(img.select(['VH'])).rename(['VH']);
    ratio = vh.divide(vv).rename(['ratio']);
    return ee.Image(ee.Image.cat(vv,vh,ratio).copyProperties(img,['system:time_start'])).clip(geom).copyProperties(img);


def toNatural(img):
    """Function to convert from dB to natural"""
    return ee.Image(10.0).pow(img.select(0).divide(10.0));
		
def toDB(img):
    """ Function to convert from natural to dB """
    return ee.Image(img).log10().multiply(10.0);


def toBands(collection):
    
    def createStack(img,prev):
	return ee.Image(prev).addBands(img)
    
    stack = ee.Image(collection.iterate(createStack,ee.Image(1)))
    stack = stack.select(ee.List.sequence(1, stack.bandNames().size().subtract(1)));
    return stack;



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



