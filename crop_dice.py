import os

# Settings
CAPTURE_DIR = 'captured_data/test2/'
CAPTURE_EXT = '.jpg'
CAPTURES_RUN_DIR = os.path.join(CAPTURE_DIR, '20181214_144327')


###################################################################################################
import numpy as np
import cv2
import datetime

KEY_UP	  = 2490368
KEY_DOWN  = 2621440
KEY_RIGHT = 2555904
KEY_LEFT  = 2424832

###################################################################################################

def readCaptureRunImage(index):
	file = "{:06d}{}".format(index, CAPTURE_EXT)
	return cv2.imread(os.path.join(CAPTURES_RUN_DIR, file))

###################################################################################################

cv2.namedWindow('main1', cv2.WINDOW_AUTOSIZE)

empty0 = cv2.imread(os.path.join(CAPTURE_DIR, "empty0.jpg"))
empty1 = cv2.imread(os.path.join(CAPTURE_DIR, "empty1.jpg"))
bowImage = cv2.imread(os.path.join(CAPTURE_DIR, "bow2.jpg"))

#empty0 = cv2.resize(empty0, (150, 102))
#empty0 = cv2.cvtColor(empty0, cv2.COLOR_BGR2HSV)
#empty0 = cv2.blur(empty0, (10, 10))

captureIndex = 0
offsetRange = 0

while (cv2.getWindowProperty('main1', 0) >= 0):	
	captureImage = readCaptureRunImage(captureIndex)
	#captureImage = cv2.resize(captureImage, (150, 102))
	#captureImage = cv2.cvtColor(captureImage, cv2.COLOR_BGR2HSV)
	
	# Variance version
	#empty0 = empty0.astype('float')
	#filterSize = (15, 15)
	#moment1 = cv2.boxFilter(empty0, -1, filterSize)
	#moment2 = cv2.boxFilter(cv2.multiply(empty0, empty0), -1, filterSize)
	#sigma = cv2.sqrt(cv2.subtract(moment2, cv2.multiply(moment1, moment1)))
	#sigma = sigma * 4	
	#minc = cv2.subtract(moment1, sigma).astype('uint8');
	#maxc = cv2.add(moment1, sigma).astype('uint8');
	
	# Simple color bounding box	
	#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
	#minc = cv2.erode(empty0, kernel)
	#maxc = cv2.dilate(empty0, kernel)
	#highPass = cv2.compare(captureImage, minc, cv2.CMP_GE)
	#lowPass = cv2.compare(captureImage, maxc, cv2.CMP_LE)	
	#highPass = cv2.min(highPass[:,:,0], cv2.min(highPass[:,:,1], highPass[:,:,2]))
	#lowPass = cv2.min(lowPass[:,:,0], cv2.min(lowPass[:,:,1], lowPass[:,:,2]))	
	#diffAccum = cv2.multiply(highPass, lowPass)
	
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
	
	height, width, channels = captureImage.shape
	diffAccum = np.zeros((height, width), np.uint8)
	diffAccum[:,:] = 255
	for xOffset in range(-offsetRange, offsetRange+1):
		for yOffset in range(-offsetRange, offsetRange+1):
			M = np.float32([[1, 0, xOffset],[0, 1, yOffset]])
			offsetImage = cv2.warpAffine(empty0, M , (width, height))
			diff = cv2.absdiff(captureImage, offsetImage)
			
			#diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
			diff = cv2.add(diff[:,:,0], cv2.add(diff[:,:,1], diff[:,:,2]))
			ret,diff = cv2.threshold(diff, 8, 255, cv2.THRESH_BINARY)
			
			diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel, iterations=20)
			diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel, iterations=3)
			
			diffAccum = cv2.min(diff, diffAccum)
	
	#diffAccum = cv2.cvtColor(diffAccum, cv2.COLOR_BGR2GRAY)
	#ret,diffAccum = cv2.threshold(diffAccum, 1, 255, cv2.THRESH_BINARY)
	
	#diffAccum = cv2.morphologyEx(diffAccum, cv2.MORPH_OPEN, kernel, iterations=2)
	#diffAccum = cv2.morphologyEx(diffAccum, cv2.MORPH_CLOSE, kernel, iterations=20)
	
	diffAccum = cv2.bitwise_and(captureImage,captureImage,mask = diffAccum)
	
	
	
	#diffAccum[:,:,0] = 0	
	#diffAccum = cv2.max(cv2.max(diffAccum[:,:,0], diffAccum[:,:,1]), diffAccum[:,:,2])	
	#captureImage = cv2.cvtColor(captureImage, cv2.COLOR_BGR2HSV)
	#diffAccum = cv2.inRange(captureImage, (170, 150, 100), (200, 255, 255))
	#diffAccum = cv2.inRange(diffAccum, (10, 10, 0), (255, 255, 255))	
	#diffAccum = cv2.blur(diffAccum, (10, 10))	
	#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	#diffAccum = cv2.morphologyEx(diffAccum, cv2.MORPH_OPEN, kernel, iterations=5)	
	#ret,diffAccum = cv2.threshold(diffAccum, 20, 255, cv2.THRESH_BINARY)	
	#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))	
	#diffAccum = cv2.morphologyEx(diffAccum, cv2.MORPH_CLOSE, kernel, iterations=10)
	
	
	
	
	#diffAccum = cv2.cvtColor(diffAccum, cv2.COLOR_BGR2GRAY)
	#ret,diffAccum = cv2.threshold(diffAccum, 50, 255, cv2.THRESH_BINARY)
	
	display = diffAccum
	
	# find the keypoints and descriptors with SIFT
	#orb = cv2.ORB_create(edgeThreshold = 2)
	#img1 = cv2.cvtColor(captureImage, cv2.COLOR_BGR2GRAY) # queryImage
	#img2 = cv2.cvtColor(bowImage, cv2.COLOR_BGR2GRAY) # trainImage
	#kp1, des1 = orb.detectAndCompute(img1,None)
	#kp2, des2 = orb.detectAndCompute(img2,None)
	#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	#matches = bf.match(des1, des2)
	#matches = sorted(matches, key = lambda x:x.distance)
	
	#print(matches[0].distance)
	
	# Draw first 10 matches.
	#img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:5],None, flags=2)
	
	#display = img3
	
	cv2.imshow('main1', display)
	key = cv2.waitKeyEx(0)
	if (key >= 0):	
		if key == KEY_RIGHT:
			captureIndex += 2
		elif key == KEY_LEFT:
			if (captureIndex > 0):
				captureIndex -= 2;
		elif key == KEY_DOWN:
			offsetRange += 1
		elif key == KEY_UP:
			if (offsetRange > 0):
				offsetRange -= 1;

cv2.destroyAllWindows()



###################################################################################################



