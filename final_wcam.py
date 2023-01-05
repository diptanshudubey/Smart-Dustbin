

import argparse
from pyzbar.pyzbar import decode
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


from PIL import Image
import pytesseract




def getNumberPlate(image, model):
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


    #image = cv2.imread(image_path)
    detections = detector.detect(image)
    
    image, crop_img , final_catagory = utils.visualize(image, detections)
    print(final_catagory)
    
    if len(detections) > 0:
        return image, crop_img
        
    return image, None





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
    if Key == 27:
        cv2.destroyAllWindows()
        break
    elif Key & 0xFF == ord('C') or Key & 0xFF == ord('c'):
        print("Capture True")
        
        #image_path = "20220323_085623(0).jpg"

        #image = cv2.imread(image_path)
        #cv2.imshow('Captured Image0', frame)
        image = frame
        finalimage, numberplate = getNumberPlate(image,'recycleableitems.tflite3')
        cv2.imshow('Captured Image1', finalimage)




#if not numberplate is None:
#    cv2.imshow('Number plate', numberplate)
#    img_p = cv2.cvtColor(numberplate, cv2.COLOR_BGR2RGB)
           
           
#while True:
#    key = cv2.waitKey(1) & 0xFF   
#    if key == ord("q") or key == ord("Q"):
#        break
                
#cv2.destroyAllWindows()
                
    

    
    
    
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    