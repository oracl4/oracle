import cv2 as cv
from openalpr import Alpr
import pprint

# Load the ALPR Module
alpr = Alpr("id", "/usr/share/openalpr/config/openalpr.defaults.conf", "/usr/share/openalpr/runtime_data")

# Open Configuration Files
# sudo subl /usr/share/openalpr/config/openalpr.defaults.conf
# sudo subl /usr/share/openalpr/runtime_data/config/id.conf
# sudo subl /usr/share/openalpr/runtime_data/

# Check the Module
if not alpr.is_loaded():
    print("Error loading OpenALPR")
    sys.exit(1)

# Camera Prewarp
prewarp = "planar, 759.000000,1024.000000,0.000050,0.000100,0.090000,1.000000,1.025000,0.000000,0.000000"

# Set the Parameter for ALPR
alpr.set_top_n(10)
# alpr.set_prewarp(prewarp)
alpr.set_default_region("jkt")

# Image Input
plate_image = cv.imread('res/b3052smd.jpg')

# Resize the Image
scale_percent = 30
width = int(plate_image.shape[1] * scale_percent / 100)
height = int(plate_image.shape[0] * scale_percent / 100)

# dsize
dsize = (width, height)

# Transform
plate_image = cv.resize(plate_image, dsize)

# Plate Recognition
LPR_Result = alpr.recognize_ndarray(plate_image)

# Print all result
pprint.pprint(LPR_Result)

# Check the Result is Empty or Not
if(LPR_Result['results'] != []):

	Plate_Result = LPR_Result['results'][0]

	return_plate = Plate_Result['plate']
	return_confidence = Plate_Result['confidence']
	return_coordinate = Plate_Result['coordinates']

	# Convert coordinates into bounding box
	top_left = return_coordinate[0]
	top_right = return_coordinate[1]
	bot_right = return_coordinate[2]
	bot_left = return_coordinate[3]

	x1=min(top_left['x'], top_right['x'], bot_right['x'], bot_left['x'])
	y1=min(top_left['y'], top_right['y'], bot_right['y'], bot_left['y'])
	x2=max(top_left['x'], top_right['x'], bot_right['x'], bot_left['x'])
	y2=max(top_left['y'], top_right['y'], bot_right['y'], bot_left['y'])

	cv.rectangle(plate_image, (x1,y1), (x2,y2), (0, 255, 0), 2)

	# Print the Result
	print("License Plate Found")
	print("  %s %s %s %.2f" % ("ALPR Result :", return_plate, "| Confidence :", return_confidence))

else:
	print("No License Plate Found in the Image")

# Shut Down the ALPR
alpr.unload()

cv.imshow('Plate LPR Result', plate_image)
cv.waitKey(0)
cv.destroyAllWindows()