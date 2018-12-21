# Settings
CAPTURE_NAME = 'test2'
RUN_NAME = '20181214_144327'
INPUT_EXT = '.jpg'
OUTPUT_EXT = '.jpg'


###################################################################################################
import os
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

XWING_GREEN_DIE_HSV_RANGE = (( 60,   0,   0), ( 90, 255, 255))
XWING_RED_DIE_HSV_RANGE   = ((160, 160,   0), (180, 255, 255))
AGE_OF_WAR_DIE_HSV_RANGE  = (( 10,  10, 210), ( 24,  80, 255))

DIE_RECT_SIZE = 84

COMPARTMENT_D_RECT = (( 60, 62), (225, 450))
COMPARTMENT_C_RECT = ((210, 62), (365, 450))
#COMPARTMENT_B_RECT = ((210, 62), (365, 450)) #TODO
COMPARTMENT_A_RECT = ((510, 62), (670, 450))

###################################################################################################

# Currently returns dieC, dieD
def compute_cropped_die_images(image):
	# X-Wing green die in comparment D
	dieD = find_hsv_range_die_in_compartment(image, COMPARTMENT_D_RECT, XWING_GREEN_DIE_HSV_RANGE)
	# X-Wing red die in compartment C
	dieC = find_hsv_range_die_in_compartment(image, COMPARTMENT_C_RECT, XWING_RED_DIE_HSV_RANGE)	
	return (dieC, dieD)

def capture_imagefile_name(index):
	file = "{:06d}{}".format(index, INPUT_EXT)
	return os.path.join('captured_data', CAPTURE_NAME, RUN_NAME, file)

def capture_image_exists(index):
	return Path(capture_imagefile_name(index)).exists()

def read_capture_image(index):
	return cv2.imread(capture_imagefile_name(index))

def save_cropped_die_image(image, compartment, file_name):
	path = os.path.join('output', CAPTURE_NAME, RUN_NAME, compartment, 'cropped')
	if not os.path.exists(path):
		os.makedirs(path)
	cv2.imwrite(os.path.join(path, file_name), image)

# Mask via HSV range filtering (good for colored dice)
def compute_hsv_range_mask(image, range, cleanup = True):
	imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(imageHSV, range[0], range[1])
	
	# Simple cleanup
	if cleanup:
		kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=10)
	
	return mask
	
def compute_mask_center(mask):
	maskMoments = cv2.moments(mask, True);
	return (int(maskMoments['m10'] / maskMoments['m00']), int(maskMoments['m01'] / maskMoments['m00']))
	
# Returns (pt1, pt2) boundary points array
def centered_rect(center, size):
	halfSize = int(size / 2)
	pt1 = (center[0] - halfSize, center[1] - halfSize)
	pt2 = (pt1[0]    +     size,    pt1[1] +     size)
	return (pt1, pt2)
	
def find_hsv_range_die_in_compartment(image, compartment_rect, hsv_range, rect_size = DIE_RECT_SIZE):
	compartment_image = image[compartment_rect[0][1]:compartment_rect[1][1], compartment_rect[0][0]:compartment_rect[1][0], :]
	mask = compute_hsv_range_mask(compartment_image, hsv_range)
		
	# Sanity check if we found something
	mask_pixels = np.count_nonzero(mask)
	if mask_pixels < 50:
		raise RuntimeException('Die not found in compartment!')
	
	die_center = compute_mask_center(mask)
	die_rect = centered_rect(die_center, rect_size)
	
	# Sanity check that it covers all the data in the mask
	# NOTE: Negative numbers have special meaning in slices, so clamp them out here
	mask[max(0, die_rect[0][1]):max(0, die_rect[1][1]), max(0, die_rect[0][0]):max(0, die_rect[1][0])] = 0
	outside_mask_count = np.count_nonzero(mask)
	if outside_mask_count > 0:
		cv2.imshow('error_mask', mask)
		print('ERROR: {} pixels outside die rectangle!'.format(outside_mask_count))
		raise RuntimeException('Die mask data outside rectangle!')
	
	# Return cropped image (from original since rectangle may overlap)
	die_rect_absolute = ((die_rect[0][0]+compartment_rect[0][0],die_rect[0][1]+compartment_rect[0][1]), (die_rect[1][0]+compartment_rect[0][0],die_rect[1][1]+compartment_rect[0][1]))	
	croppedImage = image[die_rect_absolute[0][1]:die_rect_absolute[1][1], die_rect_absolute[0][0]:die_rect_absolute[1][0]]
	if (croppedImage.shape[0] != rect_size or croppedImage.shape[1] != rect_size):
		raise RuntimeException('Die rectangle invalid!')
	
	return croppedImage


def draw_hsv_range_die_rect(source_image, output_image, hsv_range, rect_size):
	mask = compute_hsv_range_mask(source_image, hsv_range)
	die_center = compute_mask_center(mask)
	die_rect = centered_rect(die_center, rect_size)
	#output_image = cv2.circle(output_image, geometricCenter, 1, (255, 0, 255), 10)
	return cv2.rectangle(output_image, die_rect[0], die_rect[1], (255, 0, 255), 2)
	
###################################################################################################

cv2.namedWindow('main1', cv2.WINDOW_AUTOSIZE)

DieData = collections.namedtuple('DieData', ['index', 'compartment', 'known_value'])

capture_index = 4
last_capture_index = -1
test_range = 0

while (cv2.getWindowProperty('main1', 0) >= 0):
	if capture_index != last_capture_index:
		if capture_image_exists(capture_index):
			capture_image = read_capture_image(capture_index)
			print("Processing capture {}".format(capture_index))
			dieC, dieD = compute_cropped_die_images(capture_image)
		else:
			print("Capture {} not found!".format(capture_index))
		last_capture_index = capture_index
		
		display = np.concatenate((dieD, dieC), 1)

		# DEBUG
		#display = capture_image
		#display = cv2.rectangle(display, COMPARTMENT_A_RECT[0], COMPARTMENT_A_RECT[1], (0, 255, 0), 1)
		#display = draw_hsv_range_die_rect(capture_image, display, XWING_GREEN_DIE_HSV_RANGE, DIE_RECT_SIZE)
		#display = draw_hsv_range_die_rect(capture_image, display, XWING_RED_DIE_HSV_RANGE, DIE_RECT_SIZE)
		#display = draw_hsv_range_die_rect(capture_image, display, AGE_OF_WAR_DIE_HSV_RANGE, DIE_RECT_SIZE)
	
	#test_hsv_range = ((10, 10, 210), (24+test_range, 80, 255))
	#display = compute_hsv_range_mask(capture_image, test_hsv_range, True)
		
	cv2.imshow('main1', display)	
	
	key = cv2.waitKeyEx(10)
	if (key >= 0):
		if key == KEY_RIGHT:
			capture_index += 1
		elif key == KEY_LEFT:
			if (capture_index > 0):
				capture_index -= 1;
		elif key == KEY_DOWN:
			test_range += 1
			print(test_range)
		elif key == KEY_UP:
			test_range -= 1
			print(test_range)
		elif key == ord(' '):
			# Process entire directory batch-style
			file_list = Path(os.path.join('captured_data', CAPTURE_NAME, RUN_NAME)).glob('*' + INPUT_EXT)
			for file in file_list:
				file_name = os.path.basename(file)
				print("Processing {}".format(file))
				batch_image = cv2.imread(str(file))
				batch_die_c, batch_die_d = compute_cropped_die_images(batch_image)
				save_cropped_die_image(batch_die_c, 'C', file_name)
				save_cropped_die_image(batch_die_d, 'D', file_name)

cv2.destroyAllWindows()


###################################################################################################


# find the keypoints and descriptors with ORB
#orb = cv2.ORB_create(edgeThreshold = 2)
#img1 = cv2.cvtColor(capture_image, cv2.COLOR_BGR2GRAY) # queryImage
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
#highPass = cv2.compare(capture_image, minc, cv2.CMP_GE)
#lowPass = cv2.compare(capture_image, maxc, cv2.CMP_LE)	
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
#height, width, channels = capture_image.shape
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
#		diff = cv2.absdiff(capture_image, offsetImage)
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
#diffAccum = cv2.bitwise_and(capture_image,capture_image,mask = diffAccum)
