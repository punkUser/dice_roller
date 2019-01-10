# Settings
CAPTURE_DIR = 'captured_data/test4/'
CAPTURE_EXT = '.jpg'

CAPTURE_BOX_START = (90, 90)
CAPTURE_BOX_END = (840, 600)
ARDUINO_PORT = 'COM7'



###################################################################################################
import numpy as np
import cv2
import serial
import os
from pathlib import Path
import datetime

COMMAND_NONE       = 0
COMMAND_UP         = 1
COMMAND_DOWN       = 2
COMMAND_LOAD       = 3
COMMAND_CYCLE      = 4
COMMAND_CYCLE_DONE = 5
COMMAND_RANGE_TEST_UP = 6
COMMAND_RANGE_TEST_DOWN = 7
COMMAND_RANGE_TEST_VALUE = 8

KEY_UP    = 2490368
KEY_DOWN  = 2621440
KEY_RIGHT = 2555904
KEY_LEFT  = 2424832

###################################################################################################

class ArduinoSerial:
	def __init__(self):
		# NOTE: python -m serial.tools.list_ports
		# Non-blocking mode (read can return 0)
		self.m_serial = serial.Serial(ARDUINO_PORT, timeout = 0)
		#self.m_serial.write(b'180a')
	
	def __del__(self):
		self.m_serial.close();

	def write(self, command):
		self.m_serial.write(command.to_bytes(1, byteorder='little'))

	def readAvailable(self):
		return self.m_serial.in_waiting > 0
		
	def read(self):
		return int.from_bytes(self.m_serial.read(1), byteorder='little')


def saveCroppedFrame(frame, path, file):
	# TODO: Create unique path with date/time in it to avoid ever stomping anything
	if not os.path.exists(path):
		os.makedirs(path)
		
	fileName = os.path.join(path, file)
	print(fileName)

	croppedFrame = frame[CAPTURE_BOX_START[1]:CAPTURE_BOX_END[1], CAPTURE_BOX_START[0]:CAPTURE_BOX_END[0]]	
	cv2.imwrite(fileName, croppedFrame)
	

###################################################################################################

arduinoSerial = ArduinoSerial();

cap = cv2.VideoCapture(cv2.CAP_DSHOW + 0)
#cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
#cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
#cap.set(cv2.CAP_PROP_EXPOSURE, -6.0)
#cap.set(cv2.CAP_PROP_FOCUS, 45)

cv2.namedWindow('main1', cv2.WINDOW_AUTOSIZE)

captureIndex = 0
manualIndex = 0
uniqueCaptureDir = os.path.join(CAPTURE_DIR, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

while (cv2.getWindowProperty('main1', 0) >= 0):
	ret, frame = cap.read()
	
	display_image = frame.copy()
	cv2.rectangle(display_image, CAPTURE_BOX_START, CAPTURE_BOX_END, (0,255,0), 1, 8, 0)
	
	cv2.imshow('main1', display_image)
	key = cv2.waitKeyEx(1)
	if (key >= 0):
		#print(key)
	
		if key == ord('s'):
			while True:
				manualFileName = "manual_{:06d}{}".format(manualIndex, CAPTURE_EXT)
				if (not Path(os.path.join(CAPTURE_DIR, manualFileName)).exists()):
					break
				manualIndex += 1
			
			saveCroppedFrame(frame, CAPTURE_DIR, manualFileName)
		elif key == KEY_UP:
			arduinoSerial.write(COMMAND_UP)
		elif key == KEY_DOWN:
			arduinoSerial.write(COMMAND_DOWN)
		elif key == KEY_RIGHT or key == KEY_LEFT:
			arduinoSerial.write(COMMAND_LOAD)
		elif key == ord(' '):
			arduinoSerial.write(COMMAND_CYCLE)
		elif key == ord('+'):
			arduinoSerial.write(COMMAND_RANGE_TEST_UP)
		elif key == ord('-'):
			arduinoSerial.write(COMMAND_RANGE_TEST_DOWN)
			
	# Handle incoming commands
	if (arduinoSerial.readAvailable()):
		command = arduinoSerial.read();
		#print(command)
		if (command == COMMAND_CYCLE_DONE):
			saveCroppedFrame(frame, uniqueCaptureDir, "{:06d}{}".format(captureIndex, CAPTURE_EXT))
			captureIndex += 1
			
			# Auto-continue with next cycle
			arduinoSerial.write(COMMAND_CYCLE)
		
		elif (command == COMMAND_RANGE_TEST_VALUE):
			# Next value to come is the value
			while (not arduinoSerial.readAvailable()):
				pass
			value = arduinoSerial.read();	# NOTE: Divided by 10 so we can get up to 2550 in 1 byte
			print("Range test value: {}ms".format(value * 10))
	
cap.release()
cv2.destroyAllWindows()



###################################################################################################



