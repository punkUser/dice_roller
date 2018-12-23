# Settings (need to edit compute_cropped_die_images below for now as well!)
CAPTURE_DIR = 'captured_data/test4/20181222_113132/'
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

XWING_GREEN_DIE_HSV_RANGE = (( 60,  30,  60), ( 90, 255, 255))
XWING_RED_DIE_HSV_RANGE   = ((150,  80,  60), (180, 255, 255))

BLUE_CASINO_DIE_HSV_RANGE = (( 90, 120,  50), (120, 255, 255))
WHITE_DOTS_HSV_RANGE      = ((  0,   0, 200), (255,  30, 255))

XWING_DIE_RECT_SIZE = 84
CASINO_DIE_RECT_SIZE = 100

COMPARTMENT_A_RECT = (( 70, 62), (225, 450))
COMPARTMENT_B_RECT = ((220, 62), (375, 450))
COMPARTMENT_C_RECT = ((380, 62), (535, 450))
COMPARTMENT_D_RECT = ((525, 62), (685, 450))

###################################################################################################

# Currently returns dieA, dieB, dieC, dieD
# NOTE: Each may return "None" if no die is found
def compute_cropped_die_images(image):
	dieA = crop_hsv_range_die_in_compartment(image, COMPARTMENT_A_RECT, [WHITE_DOTS_HSV_RANGE], CASINO_DIE_RECT_SIZE)
	dieB = crop_hsv_range_die_in_compartment(image, COMPARTMENT_B_RECT, [XWING_GREEN_DIE_HSV_RANGE], XWING_DIE_RECT_SIZE)
	dieC = crop_hsv_range_die_in_compartment(image, COMPARTMENT_C_RECT, [XWING_RED_DIE_HSV_RANGE], XWING_DIE_RECT_SIZE)
	dieD = crop_hsv_range_die_in_compartment(image, COMPARTMENT_D_RECT, [WHITE_DOTS_HSV_RANGE], CASINO_DIE_RECT_SIZE)
	return (dieA, dieB, dieC, dieD)

def capture_imagefile_name(index):
	file = "{:06d}{}".format(index, INPUT_EXT)
	return os.path.join(CAPTURE_DIR, file)

def capture_image_exists(index):
	return Path(capture_imagefile_name(index)).exists()

def read_capture_image(index):
	return cv2.imread(capture_imagefile_name(index))

def save_cropped_die_image(image, compartment, file_name):
	path = os.path.join('output', CAPTURE_DIR, compartment, 'cropped')
	if not os.path.exists(path):
		os.makedirs(path)
	cv2.imwrite(os.path.join(path, file_name), image)

# Mask via HSV range filtering (good for colored dice)
def compute_hsv_range_mask(image, ranges, cleanup = True, open_iterations = 2, close_iterations = 5):
	imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	mask = np.zeros(image.shape[0:2], dtype = np.uint8)
	for range in ranges:
		range_mask = cv2.inRange(imageHSV, range[0], range[1])
		mask = cv2.max(mask, range_mask)
	
	# Simple cleanup
	if cleanup:
		kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iterations)
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
	
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

# Returns die rectangle
def find_hsv_range_die_in_compartment(image, compartment_rect, hsv_ranges, rect_size):
	compartment_image = image[compartment_rect[0][1]:compartment_rect[1][1], compartment_rect[0][0]:compartment_rect[1][0], :]
	mask = compute_hsv_range_mask(compartment_image, hsv_ranges)
		
	# Sanity check if we found something
	mask_pixels = np.count_nonzero(mask)
	if mask_pixels < 50:
		return ((0, 0), (0, 0))
	
	die_center = compute_mask_center(mask)
	die_rect = centered_rect(die_center, rect_size)
	
	# Sanity check that it covers all the data in the mask
	# NOTE: Negative numbers have special meaning in slices, so clamp them out here
	mask[max(0, die_rect[0][1]):max(0, die_rect[1][1]), max(0, die_rect[0][0]):max(0, die_rect[1][0])] = 0
	outside_mask_count = np.count_nonzero(mask)
	if outside_mask_count > 20:
		# Allow to fall through since this is often a non-critical error
		#cv2.imshow('error_mask', mask)
		print('ERROR: {} pixels outside die rectangle!'.format(outside_mask_count))
	
	die_rect_absolute = ((die_rect[0][0]+compartment_rect[0][0],die_rect[0][1]+compartment_rect[0][1]), (die_rect[1][0]+compartment_rect[0][0],die_rect[1][1]+compartment_rect[0][1]))
	
	return die_rect_absolute

# Returns "None" if die rect is invalid (die not found)
def crop_hsv_range_die_in_compartment(image, compartment_rect, hsv_ranges, rect_size):
	die_rect = find_hsv_range_die_in_compartment(image, compartment_rect, hsv_ranges, rect_size)
		
	cropped_image = image[die_rect[0][1]:die_rect[1][1], die_rect[0][0]:die_rect[1][0]]
	if (cropped_image.shape[0] != rect_size or cropped_image.shape[1] != rect_size):
		return None
	
	return cropped_image
	
def draw_hsv_range_die_rect(output_image, source_image, compartment_rect, hsv_ranges, rect_size):
	die_rect = find_hsv_range_die_in_compartment(source_image, compartment_rect, hsv_ranges, rect_size)
	return cv2.rectangle(output_image, die_rect[0], die_rect[1], (255, 0, 255), 1)
	
###################################################################################################

def concat_images(images):
	total_width = 0
	max_height = 0
	for image in images:
		if image is not None:
			total_width += image.shape[1]
			max_height = max(max_height, image.shape[0])
	
	new_image = np.zeros(shape=(max_height, total_width, 3), dtype=np.uint8)
	x = 0
	for image in images:
		if image is not None:
			width = image.shape[1]
			height = image.shape[0]
			new_image[:height,x:x+width,:] = image
			x += width
		
	return new_image

cv2.namedWindow('main1', cv2.WINDOW_AUTOSIZE)

DieData = collections.namedtuple('DieData', ['index', 'compartment', 'known_value'])

capture_index = 100
last_capture_index = -1
test_range = 0
tuning_ranges = False

while (cv2.getWindowProperty('main1', 0) >= 0):
	if capture_index != last_capture_index:
		if capture_image_exists(capture_index):
			capture_image = read_capture_image(capture_index)
			print("Loaded capture index {}".format(capture_index))
		else:
			print("Capture {} not found!".format(capture_index))
		last_capture_index = capture_index

	if tuning_ranges:
		test_hsv_range = (( 60, test_range,  60), ( 90, 255, 255))
		display = compute_hsv_range_mask(capture_image, [test_hsv_range], False)
	else:
		rect_display = capture_image.copy()
		rect_display = cv2.rectangle(rect_display, COMPARTMENT_A_RECT[0], COMPARTMENT_D_RECT[1], (0, 255, 0), 1)
		rect_display = cv2.rectangle(rect_display, COMPARTMENT_B_RECT[0], COMPARTMENT_C_RECT[1], (0, 255, 0), 1)
		rect_display = cv2.rectangle(rect_display, COMPARTMENT_C_RECT[0], COMPARTMENT_B_RECT[1], (0, 255, 0), 1)
		rect_display = cv2.rectangle(rect_display, COMPARTMENT_D_RECT[0], COMPARTMENT_A_RECT[1], (0, 255, 0), 1)
		rect_display = draw_hsv_range_die_rect(rect_display, capture_image, COMPARTMENT_A_RECT, [WHITE_DOTS_HSV_RANGE], CASINO_DIE_RECT_SIZE)
		rect_display = draw_hsv_range_die_rect(rect_display, capture_image, COMPARTMENT_B_RECT, [XWING_GREEN_DIE_HSV_RANGE], XWING_DIE_RECT_SIZE)
		rect_display = draw_hsv_range_die_rect(rect_display, capture_image, COMPARTMENT_C_RECT, [XWING_RED_DIE_HSV_RANGE], XWING_DIE_RECT_SIZE)
		rect_display = draw_hsv_range_die_rect(rect_display, capture_image, COMPARTMENT_D_RECT, [WHITE_DOTS_HSV_RANGE], CASINO_DIE_RECT_SIZE)
		
		dieA, dieB, dieC, dieD = compute_cropped_die_images(capture_image)
		
		if dieB is None:
			print("Missing B!")
		
		display = concat_images([rect_display, dieA, dieB, dieC, dieD])
	
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
		elif key == ord('t'):
			tuning_ranges = not tuning_ranges
		elif key == ord(' '):
			# Process entire directory batch-style
			missing_count = [0, 0, 0, 0]		# ABCD
			
			file_list = Path(os.path.join(CAPTURE_DIR)).glob('*' + INPUT_EXT)
			for file in file_list:
				file_name = os.path.basename(file)
				print("Processing {}".format(file))
				batch_image = cv2.imread(str(file))
				batch_die_a, batch_die_b, batch_die_c, batch_die_d = compute_cropped_die_images(batch_image)
				# Skip missing dice for now
				if batch_die_a is not None:
					save_cropped_die_image(batch_die_a, 'A', file_name)
				else:
					missing_count[0] += 1
				
				if batch_die_b is not None:
					save_cropped_die_image(batch_die_b, 'B', file_name)
				else:
					missing_count[1] += 1
					
				if batch_die_c is not None:
					save_cropped_die_image(batch_die_c, 'C', file_name)
				else:
					missing_count[2] += 1
					
				if batch_die_d is not None:
					save_cropped_die_image(batch_die_d, 'D', file_name)
				else:
					missing_count[3] += 1
			
			print("Missing dice (A, B, C, D): {}".format(missing_count))

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
