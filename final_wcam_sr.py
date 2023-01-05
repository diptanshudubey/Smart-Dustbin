

import argparse

from PIL import Image

import sys
import time

import cv2
from object_detector import ObjectDetector
from object_detector import ObjectDetectorOptions
import utils
import serial
import time

#python fianl_code.py --image=onlynumber.jpg --port=COM11 --limit=30


_NB = "Non biodegradable"
_BA = "Biodegradable"

class Garbage():
    name = ""
    cat = ""


product_catagories = []

g1 = Garbage()
g1.name = "ezee"
g1.cat = _NB

g2 = Garbage()
g2.name = "dabur_red"
g2.cat = _NB


g3 = Garbage()
g3.name = "appy_fizz"
g3.cat = _NB


g4 = Garbage()
g4.name = "wooden_spoon"
g4.cat = _BA

g5 = Garbage()
g5.name = "potato"
g5.cat = _BA

g6 = Garbage()
g6.name = "onion"
g6.cat = _BA

    
BAUD = 9600   




import serial.tools.list_ports
ports = serial.tools.list_ports.comports()
print("Available ports are")
for port, desc, hwid in sorted(ports):
        #print("{}: {} [{}]".format(port, desc, hwid))
        print("{}: {}".format(port, desc))
        
        
parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
                
parser.add_argument(
            '--port',
            help='COM port',
            required=True,
            default='COM8')

parser.add_argument(
                '--thress',
                help='Thres Hold',
                required=False,
                default=0.3)
                
parser.add_argument(
                '--cam',
                help='camera number',
                required=False,
                default=1)

args = parser.parse_args()
    
arg_score_threshold = args.thress
 
serialPort = serial.Serial(args.port, BAUD, timeout=1)
if serialPort.isOpen():
    print(serialPort.name + ' is open...')






# Initialize the object detection model
options = ObjectDetectorOptions(
    num_threads=4,
    score_threshold=arg_score_threshold,
    max_results=3,
    enable_edgetpu= bool(False))
  
detector = ObjectDetector(model_path='recycleableitems.tflite3', options=options)

cam = int(args.cam)

webcam = cv2.VideoCapture(1)
while True:

    # Loop until the camera is working
    rval = False
    while(not rval):
        # Put the image from the webcam into 'frame'
        (rval, frame) = webcam.read()
        if(not rval):
            print("Failed to open webcam. Trying again...")

    cv2.imshow('Click & Press C to capture', frame)


    Key = cv2.waitKey(1)
    if Key == 27 or Key & 0xFF == ord('Q') or Key & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    
    elif Key & 0xFF == ord('C') or Key & 0xFF == ord('c'):
        print("Capture True")

        image = frame
        detections = detector.detect(image)
        image, crop_img,final_catagory = utils.visualize(image, detections)
        cv2.imshow('Result', image)
        print('final_catagory',final_catagory)
        
        category = None
        
        if g1.name == final_catagory:
           print(g1.cat)
           category = g1.cat
        
        elif g2.name == final_catagory:
           print(g2.cat)
           category = g2.cat
        
        elif g3.name == final_catagory:
           print(g3.cat)
           category = g3.cat
        
        elif g4.name == final_catagory:
           print(g4.cat)
           category = g4.cat
           
        elif g5.name == final_catagory:
           print(g5.cat)
           category = g5.cat
        
        elif g6.name == final_catagory:
           print(g6.cat)
           category = g6.cat
        
        
        if not category is None:
            print("category", category[0])
            serialPort.write(str('\r'+category[0]+'\n').encode('ascii'))
        



                
    

    
    
    
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    