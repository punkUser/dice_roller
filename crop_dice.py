import os
import numpy as np
import cv2
import datetime
import collections
import os
from pathlib import Path
import die_types

# Settings
CAPTURE_DIR = 'captured_data/xr10_xr11_xr12_xr13/20190131_134655/'
INPUT_EXT = '.jpg'
OUTPUT_EXT = '.jpg'

# Compartments ABCD; should match the types in die_types.py
DIE_TYPES = ["xwing_red", "xwing_red", "xwing_red", "xwing_red"]


###################################################################################################

KEY_UP	  = 2490368
KEY_DOWN  = 2621440
KEY_RIGHT = 2555904
KEY_LEFT  = 2424832

COMPARTMENT_A_RECT = (( 70, 62), (225, 450))
COMPARTMENT_B_RECT = ((210, 62), (365, 450))
COMPARTMENT_C_RECT = ((360, 62), (515, 450))
COMPARTMENT_D_RECT = ((505, 62), (665, 450))

###################################################################################################

# Currently returns dieA, dieB, dieC, dieD
# NOTE: Each may return "None" if no die is found
def compute_cropped_die_images(image):
	dieA = crop_hsv_range_die_in_compartment(image, COMPARTMENT_A_RECT, die_types.params[DIE_TYPES[0]]["hsv_ranges"], die_types.params[DIE_TYPES[0]]["rect_size"])
	dieB = crop_hsv_range_die_in_compartment(image, COMPARTMENT_B_RECT, die_types.params[DIE_TYPES[1]]["hsv_ranges"], die_types.params[DIE_TYPES[1]]["rect_size"])
	dieC = crop_hsv_range_die_in_compartment(image, COMPARTMENT_C_RECT, die_types.params[DIE_TYPES[2]]["hsv_ranges"], die_types.params[DIE_TYPES[2]]["rect_size"])
	dieD = crop_hsv_range_die_in_compartment(image, COMPARTMENT_D_RECT, die_types.params[DIE_TYPES[3]]["hsv_ranges"], die_types.params[DIE_TYPES[3]]["rect_size"])
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
	# TODO: probably want some sort of test ID in the file name as well (just simple index)
	cv2.imwrite(os.path.join(path, "{}_{}".format(compartment, file_name)), image)

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

def compare_images(image1, image2):
	delta = cv2.absdiff(image1, image2)
	return np.mean(delta)
	
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
	fraction_outside = outside_mask_count / mask_pixels
	if fraction_outside > 0.15:
		# Allow to fall through since this is often a non-critical error
		#cv2.imshow('error_mask', mask)
		print('WARNING: {}% pixels outside die rectangle!'.format(fraction_outside * 100.0))
	
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

capture_index = 0
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
		#test_hsv_range = ((  0,   0, 200), (255,  30+test_range, 255))
		#test_hsv_range = ((150,  80,  60), (180, 255, 255))
		test_hsv_range = ((0,  140, 60+test_range), (10, 255, 255))
		test_hsv_range_2 = ((150,  80, 60), (180, 255, 255))
		display = compute_hsv_range_mask(capture_image, [test_hsv_range, test_hsv_range_2], False)
	else:
		rect_display = capture_image.copy()
		rect_display = cv2.rectangle(rect_display, COMPARTMENT_A_RECT[0], COMPARTMENT_D_RECT[1], (0, 255, 0), 1)
		rect_display = cv2.rectangle(rect_display, COMPARTMENT_B_RECT[0], COMPARTMENT_C_RECT[1], (0, 255, 0), 1)
		rect_display = cv2.rectangle(rect_display, COMPARTMENT_C_RECT[0], COMPARTMENT_B_RECT[1], (0, 255, 0), 1)
		rect_display = cv2.rectangle(rect_display, COMPARTMENT_D_RECT[0], COMPARTMENT_A_RECT[1], (0, 255, 0), 1)
		rect_display = draw_hsv_range_die_rect(rect_display, capture_image, COMPARTMENT_A_RECT, die_types.params[DIE_TYPES[0]]["hsv_ranges"], die_types.params[DIE_TYPES[0]]["rect_size"])
		rect_display = draw_hsv_range_die_rect(rect_display, capture_image, COMPARTMENT_B_RECT, die_types.params[DIE_TYPES[1]]["hsv_ranges"], die_types.params[DIE_TYPES[1]]["rect_size"])
		rect_display = draw_hsv_range_die_rect(rect_display, capture_image, COMPARTMENT_C_RECT, die_types.params[DIE_TYPES[2]]["hsv_ranges"], die_types.params[DIE_TYPES[2]]["rect_size"])
		rect_display = draw_hsv_range_die_rect(rect_display, capture_image, COMPARTMENT_D_RECT, die_types.params[DIE_TYPES[3]]["hsv_ranges"], die_types.params[DIE_TYPES[3]]["rect_size"])
		
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
			total_count = 0
			last_image = None
			file_list = Path(os.path.join(CAPTURE_DIR)).glob('*' + INPUT_EXT)
			for i, file in enumerate(file_list):
				file_name = os.path.basename(file)
				if (i % 1000 == 0):
					print("Processing {}".format(file))
				total_count += 1
								
				batch_image = cv2.imread(str(file))				
				if last_image is not None:
					last_image_delta = compare_images(last_image, batch_image)
					if (last_image_delta < 5):
						print("WARNING: Potentially duplicate image detected {}. (Delta = {})".format(file, last_image_delta))
				last_image = batch_image
				
				batch_die_a, batch_die_b, batch_die_c, batch_die_d = compute_cropped_die_images(batch_image)
				
				# Skip missing dice for now
				if batch_die_a is not None:
					save_cropped_die_image(batch_die_a, 'A', file_name)
				else:
					print("Missing compartment A in {}".format(file))
					missing_count[0] += 1
				
				if batch_die_b is not None:
					save_cropped_die_image(batch_die_b, 'B', file_name)
				else:
					print("Missing compartment B in {}".format(file))
					missing_count[1] += 1
					
				if batch_die_c is not None:
					save_cropped_die_image(batch_die_c, 'C', file_name)
				else:
					print("Missing compartment C in {}".format(file))
					missing_count[2] += 1
					
				if batch_die_d is not None:
					save_cropped_die_image(batch_die_d, 'D', file_name)
				else:
					print("Missing compartment D in {}".format(file))
					missing_count[3] += 1
			
			print("Scanned {} images. Missing dice (A, B, C, D): {}".format(total_count, missing_count))

cv2.destroyAllWindows()

