import os
import numpy as np
import cv2
import os
from pathlib import Path
import die_types
import shutil

# Settings
INPUT_DIR = 'C:/Users/Andrew/Desktop/tosort'
OUTPUT_DIR = 'C:/Users/Andrew/Desktop/sorted'
INPUT_EXT = '.jpg'

DIE_TYPE = "d8_blue"

###################################################################################################

KEY_UP	  = 2490368
KEY_DOWN  = 2621440
KEY_RIGHT = 2555904
KEY_LEFT  = 2424832

###################################################################################################

def read_capture_image(index):
	return cv2.imread(capture_imagefile_name(index))

cv2.namedWindow('main1', cv2.WINDOW_AUTOSIZE)

capture_list = list(Path(os.path.join(INPUT_DIR)).glob('*{}'.format(INPUT_EXT)))
print("Found {} files".format(len(capture_list)))
capture_index = 0
last_capture_index = -1

while (cv2.getWindowProperty('main1', 0) >= 0):
	if capture_index != last_capture_index:
		file_name = capture_list[capture_index]
		base_file_name = os.path.basename(file_name)
		
		if file_name.exists():
			capture_image = cv2.imread(str(file_name))
			capture_image = cv2.resize(capture_image, (400, 400))
			print("Loaded capture index {}".format(base_file_name))
		else:
			print("Capture {} not found!".format(base_file_name))
			
		last_capture_index = capture_index
	
	cv2.imshow('main1', capture_image)
	
	category = None
	
	key = cv2.waitKeyEx(10)
	if (key >= 0):
		if key == KEY_RIGHT:
			capture_index += 1
		elif key == KEY_LEFT:
			if (capture_index > 0):
				capture_index -= 1;
		elif key == ord('1'):
			category = "one"
		elif key == ord('2'):
			category = "two"
		elif key == ord('3'):
			category = "three"
		elif key == ord('4'):
			category = "four"
		elif key == ord('5'):
			category = "five"
		elif key == ord('6'):
			category = "six"
		elif key == ord('7'):
			category = "seven"
		elif key == ord('8'):
			category = "eight"
		
		if category is not None:
			path = os.path.join(OUTPUT_DIR, category)
			if not os.path.exists(path):
				os.makedirs(path)
			
			output_file = os.path.join(path, base_file_name)
			if (Path(output_file).exists()):
				print("Cannot categorize {} as {}: output file {} exists!".format(base_file_name, category, output_file))
			else:
				shutil.move(file_name, output_file)
				print("Categorized {} as {} ({})".format(base_file_name, category, output_file))
				capture_index = capture_index + 1			

cv2.destroyAllWindows()

