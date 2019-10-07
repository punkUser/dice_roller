# Settings
CAPTURE_DIR = 'captured_data/cxd6p9_cxd6p10_cxd6p11_cxd6p12/'
CAPTURE_EXT = '.jpg'
INITIAL_CAPTURE_INDEX = 0

CAPTURE_BOX_START = (140, 150)
CAPTURE_BOX_END = (730, 540)
ARDUINO_PORT = 'COM3'

# Set to zero or negative to disable early cycle
CYCLE_MOTION_THRESHOLD = 0.55
MAX_CYCLE_WAIT_TIME_SECONDS = 10.0



###################################################################################################
import numpy as np
import cv2
import serial
import os
from pathlib import Path
import datetime
import time

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

def cropFrame(frame):
    return frame[CAPTURE_BOX_START[1]:CAPTURE_BOX_END[1], CAPTURE_BOX_START[0]:CAPTURE_BOX_END[0]]

def saveCroppedFrame(frame, path, file):
    # TODO: Create unique path with date/time in it to avoid ever stomping anything
    if not os.path.exists(path):
        os.makedirs(path)
        
    fileName = os.path.join(path, file)
    print(fileName)

    cv2.imwrite(fileName, cropFrame(frame))

def denoise_image(image):
    image = cv2.GaussianBlur(image, (11, 11), 5)
    return image

def compare_cropped_images(image1, image2):
    image1Denoise = denoise_image(cropFrame(image1));
    image2Denoise = denoise_image(cropFrame(image2));
    delta = cv2.absdiff(image1Denoise, image2Denoise)
    return np.mean(delta)


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

captureIndex = INITIAL_CAPTURE_INDEX
manualIndex = 0
uniqueCaptureDir = os.path.join(CAPTURE_DIR, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

cycleDone = False
cycleEndTime = time.time()

lastImageCompareFrame = None
lastImageCompareTime = time.time()
lastFrameCompareDelta = 9999999999

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
            cycleDone = False
            print("UP")
        elif key == KEY_DOWN:
            arduinoSerial.write(COMMAND_DOWN)
            cycleDone = False
            print("DOWN")
        elif key == KEY_RIGHT or key == KEY_LEFT:
            arduinoSerial.write(COMMAND_LOAD)
            cycleDone = False
            print("LOAD")
        elif key == ord(' '):
            arduinoSerial.write(COMMAND_CYCLE)
            cycleDone = False
        elif key == ord('+'):
            arduinoSerial.write(COMMAND_RANGE_TEST_UP)
        elif key == ord('-'):
            arduinoSerial.write(COMMAND_RANGE_TEST_DOWN)
            
    # Handle incoming commands
    if (arduinoSerial.readAvailable()):
        command = arduinoSerial.read();
        #print(command)
        if (command == COMMAND_CYCLE_DONE):
            cycleDone = True
            cycleEndTime = time.time()
        elif (command == COMMAND_RANGE_TEST_VALUE):
            # Next value to come is the value
            while (not arduinoSerial.readAvailable()):
                pass
            value = arduinoSerial.read();   # NOTE: Divided by 10 so we can get up to 2550 in 1 byte
            print("Range test value: {}us".format(value * 10))
    
    currentTime = time.time()
    
    if (currentTime - lastImageCompareTime) > 0.5:
        if lastImageCompareFrame is not None:
            lastFrameCompareDelta = compare_cropped_images(lastImageCompareFrame, frame)
            #print(lastFrameCompareDelta)
        lastImageCompareFrame = frame
        lastImageCompareTime = currentTime
    
    # Check for next cycle
    if cycleDone:
        if ((currentTime - cycleEndTime) > MAX_CYCLE_WAIT_TIME_SECONDS) or (lastFrameCompareDelta < CYCLE_MOTION_THRESHOLD):
            saveCroppedFrame(frame, uniqueCaptureDir, "{:06d}{}".format(captureIndex, CAPTURE_EXT))
            captureIndex += 1
            
            # Auto-continue with next cycle
            arduinoSerial.write(COMMAND_CYCLE)
            cycleDone = False
    
cap.release()
cv2.destroyAllWindows()



###################################################################################################



