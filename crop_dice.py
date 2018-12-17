import os

# Settings
CAPTURE_NAME = 'test2'
RUN_NAME = '20181214_144327'
IMAGE_EXT = '.jpg'


###################################################################################################
import numpy as np
import cv2
import datetime
import collections
import os
from pathlib import Path

KEY_UP	  = 2490368
KEY_DOWN  = 2621440
KEY_RIGHT = 2555904
KEY_LEFT  = 2424832

XWING_GREEN_DIE_HSV_RANGE = (( 60,   0, 0), ( 90, 255, 255))
XWING_RED_DIE_HSV_RANGE   = ((160, 160, 0), (180, 255, 255))

DIE_RECT_SIZE = 84

COMPARTMENT_D_RECT = (( 60, 62), (225, 450))
COMPARTMENT_C_RECT = ((210, 62), (365, 450))

###################################################################################################

def captureImageFileName(index):
	file = "{:06d}{}".format(index, IMAGE_EXT)
	return os.path.join('captured_data', CAPTURE_NAME, RUN_NAME, file)

def captureImageExists(index):
	return Path(captureImageFileName(index)).exists()

def readCaptureImage(index):
	return cv2.imread(captureImageFileName(index))

def saveCroppedDieImage(image, index, compartment):
	file = "{:06d}{}".format(index, IMAGE_EXT)
	path = os.path.join('output', CAPTURE_NAME, RUN_NAME, compartment)	
	if not os.path.exists(path):
		os.makedirs(path)	
	cv2.imwrite(os.path.join(path, file), image)

# Mask via HSV range filtering (good for colored dice)
def computeHSVRangeMask(image, range):
	imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(imageHSV, range[0], range[1])
	
	# Simple cleanup
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=10)
	
	return mask
	
def computeMaskCenter(mask):
	maskMoments = cv2.moments(mask, True);
	return (int(maskMoments['m10'] / maskMoments['m00']), int(maskMoments['m01'] / maskMoments['m00']))
	
# Returns (pt1, pt2) boundary points array
def centeredRect(center, size):
	halfSize = int(size / 2)
	pt1 = (center[0] - halfSize, center[1] - halfSize)
	pt2 = (pt1[0]    +     size,    pt1[1] +     size)
	return (pt1, pt2)
	
def findHSVRangeDieInCompartment(image, compartmentRect, hsvRange, rectSize = DIE_RECT_SIZE):
	compartmentImage = image[compartmentRect[0][1]:compartmentRect[1][1], compartmentRect[0][0]:compartmentRect[1][0], :]
	mask = computeHSVRangeMask(compartmentImage, hsvRange)
	
	# Debug
	#display = cv2.bitwise_and(captureImage, captureImage, mask = mask)
	
	# Sanity check if we found something
	maskPixels = np.count_nonzero(mask)
	if maskPixels < 50:
		raise ValueException('Die not found in compartment!')
	
	dieCenter = computeMaskCenter(mask)
	dieRect = centeredRect(dieCenter, rectSize)
	
	# Sanity check that it covers all the data in the mask
	# NOTE: Negative numbers have special meaning in slices, so clamp them out here
	mask[max(0, dieRect[0][1]):max(0, dieRect[1][1]), max(0, dieRect[0][0]):max(0, dieRect[1][0])] = 0
	if np.count_nonzero(mask) > 0:
		cv2.imshow('error_mask', mask)
		print('ERROR: Die mask data outside rectangle!')
		#raise ValueException('Die mask data outside rectangle!')
	
	# Return cropped image (from original since rectangle may overlap)
	dieRectAbsolute = ((dieRect[0][0]+compartmentRect[0][0],dieRect[0][1]+compartmentRect[0][1]), (dieRect[1][0]+compartmentRect[0][0],dieRect[1][1]+compartmentRect[0][1]))	
	croppedImage = image[dieRectAbsolute[0][1]:dieRectAbsolute[1][1], dieRectAbsolute[0][0]:dieRectAbsolute[1][0]]
	if (croppedImage.shape[0] != rectSize or croppedImage.shape[1] != rectSize):
		raise ValueException('Die rectangle invalid!')
	
	return croppedImage


def drawHSVRangeDieRect(sourceImage, outputImage, hsvRange, rectSize):
	mask = computeHSVRangeMask(sourceImage, hsvRange)
	dieCenter = computeMaskCenter(mask)
	dieRect = centeredRect(dieCenter, rectSize)
	#outputImage = cv2.circle(outputImage, geometricCenter, 1, (255, 0, 255), 10)
	return cv2.rectangle(outputImage, dieRect[0], dieRect[1], (255, 0, 255), 2)
	
###################################################################################################

cv2.namedWindow('main1', cv2.WINDOW_AUTOSIZE)

DieData = collections.namedtuple('DieData', ['index', 'compartment', 'known_value'])

captureIndex = 0
autoAdvance = False

while (cv2.getWindowProperty('main1', 0) >= 0):
	captureImage = readCaptureImage(captureIndex)
	print("Processing capture {}".format(captureIndex))

	# X-Wing green die in comparment D
	dieD = findHSVRangeDieInCompartment(captureImage, COMPARTMENT_D_RECT, XWING_GREEN_DIE_HSV_RANGE)
	# X-Wing red die in compartment C
	dieC = findHSVRangeDieInCompartment(captureImage, COMPARTMENT_C_RECT, XWING_RED_DIE_HSV_RANGE)
	
	saveCroppedDieImage(dieD, captureIndex, 'D')
	saveCroppedDieImage(dieC, captureIndex, 'C')
	
	display = np.concatenate((dieD, dieC), 1)

	# DEBUG
	#display = captureImage
	#display = cv2.rectangle(display, COMPARTMENT_D_RECT[0], COMPARTMENT_D_RECT[1], (0, 255, 0), 1)
	#display = drawHSVRangeDieRect(captureImage, display, XWING_GREEN_DIE_HSV_RANGE, DIE_RECT_SIZE)
	#display = drawHSVRangeDieRect(captureImage, display, XWING_RED_DIE_HSV_RANGE, DIE_RECT_SIZE)
	
	cv2.imshow('main1', display)
	
	key = cv2.waitKeyEx(1 if autoAdvance else 0)
	if (key >= 0):
		if key == KEY_RIGHT:
			captureIndex += 1
		elif key == KEY_LEFT:
			if (captureIndex > 0):
				captureIndex -= 1;
		elif key == KEY_DOWN:
			pass
		elif key == KEY_UP:
			pass
		elif key == ord(' '):
			autoAdvance = not autoAdvance
	
	if autoAdvance:
		captureIndex += 1
		if (not captureImageExists(captureIndex)):
			autoAdvance = False
			captureIndex -= 1

cv2.destroyAllWindows()



###################################################################################################


# find the keypoints and descriptors with ORB
#orb = cv2.ORB_create(edgeThreshold = 2)
#img1 = cv2.cvtColor(captureImage, cv2.COLOR_BGR2GRAY) # queryImage
#img2 = cv2.cvtColor(bowImage, cv2.COLOR_BGR2GRAY) # trainImage
#kp1, des1 = orb.detectAndCompute(img1,None)
#kp2, des2 = orb.detectAndCompute(img2,None)
#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#matches = bf.match(des1, des2)
#matches = sorted(matches, key = lambda x:x.distance)
#img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:5],None, flags=2)

# Simple color bounding box	
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
#minc = cv2.erode(empty0, kernel)
#maxc = cv2.dilate(empty0, kernel)
#highPass = cv2.compare(captureImage, minc, cv2.CMP_GE)
#lowPass = cv2.compare(captureImage, maxc, cv2.CMP_LE)	
#highPass = cv2.min(highPass[:,:,0], cv2.min(highPass[:,:,1], highPass[:,:,2]))
#lowPass = cv2.min(lowPass[:,:,0], cv2.min(lowPass[:,:,1], lowPass[:,:,2]))	
#diffAccum = cv2.multiply(highPass, lowPass)

# Variance version
#empty0 = empty0.astype('float')
#filterSize = (15, 15)
#moment1 = cv2.boxFilter(empty0, -1, filterSize)
#moment2 = cv2.boxFilter(cv2.multiply(empty0, empty0), -1, filterSize)
#sigma = cv2.sqrt(cv2.subtract(moment2, cv2.multiply(moment1, moment1)))
#sigma = sigma * 4	
#minc = cv2.subtract(moment1, sigma).astype('uint8');
#maxc = cv2.add(moment1, sigma).astype('uint8');

# RGB distance from background image
#kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))	
#height, width, channels = captureImage.shape
#diffAccum = np.zeros((height, width), np.uint8)
#diffAccum[:,:] = 255
#for xOffset in range(-offsetRange, offsetRange+1):
#	for yOffset in range(-offsetRange, offsetRange+1):
#		M = np.float32([[1, 0, xOffset],[0, 1, yOffset]])
#		offsetImage = cv2.warpAffine(empty0, M , (width, height))
#		
#		#diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
#		
#		# RGB euclidean distance
#		diff = cv2.absdiff(captureImage, offsetImage)
#		
#		diff = np.float32(diff)
#		diff = cv2.multiply(diff, diff)
#		diff = cv2.add(diff[:,:,0], cv2.add(diff[:,:,1], diff[:,:,2]))
#		diff = cv2.sqrt(diff)
#		ret,diff = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
#		diff = np.uint8(diff)
		
#		diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel, iterations=5)
#		diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel, iterations=5)
#		diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel, iterations=10)
		#diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel, iterations=2)
		
#		diffAccum = cv2.min(diff, diffAccum)
#diffAccum = cv2.bitwise_and(captureImage,captureImage,mask = diffAccum)
