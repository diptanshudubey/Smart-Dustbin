

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

 
serialPort = serial.Serial(args.port, BAUD, timeout=1)
if serialPort.isOpen():
    print(serialPort.name + ' is open...')




def getNumberPlate(image_path, model):
    row_size = 20  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 255)  # red
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    # Initialize the object detection model
    options = ObjectDetectorOptions(
        num_threads=4,
        score_threshold=0.3,
        max_results=3,
        enable_edgetpu= bool(False))
      
    detector = ObjectDetector(model_path=model, options=options)


    image = cv2.imread(image_path)
    detections = detector.detect(image)
    
    image, crop_img = utils.visualize(image, detections)
    
    if len(detections) > 0:
        return image, crop_img
        
    return image, None

image_path = "20220323_085623(0).jpg"

image = cv2.imread(image_path)
cv2.imshow('Captured Image0', image)
finalimage, numberplate = getNumberPlate(image_path,'gross3.tflite3')
cv2.imshow('Captured Image1', finalimage)
#if not numberplate is None:
#    cv2.imshow('Number plate', numberplate)
#    img_p = cv2.cvtColor(numberplate, cv2.COLOR_BGR2RGB)
           
           
while True:
    key = cv2.waitKey(1) & 0xFF   
    if key == ord("q") or key == ord("Q"):
        break
                
cv2.destroyAllWindows()
                
    

    
    
    
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    