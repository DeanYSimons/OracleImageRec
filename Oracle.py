#+-------------------------------------------------------------------------+#
#|                                                                         |#
#|                              Sprit Reader                               |#
#|                                                                         |#
#|                            By: Dean Simons                              |#
#|                             Using YOLOV11                               |#
#|                                                                         |#
#|                       Captial Region Scrapyard                          |#
#|                             Hackathon 2025                              |#
#|                                                                         |#
#+-------------------------------------------------------------------------+#

# This script is designed to read images from a webcam and use a YOLOv11 model to detect humans in frame and react accoringly to their "Spirit".


#---------------------------------------------------------------------------#
#-------------------------------- IMPORTS ----------------------------------#
#---------------------------------------------------------------------------#

import cv2
import random
from ultralytics import YOLO
from playsound import playsound
from ultralytics.solutions.object_counter import ObjectCounter
import time


#---------------------------------------------------------------------------#
#-------------------------------- PROGRAM ----------------------------------#
#---------------------------------------------------------------------------#

print("Start of program : line 27") #Marker 


#Loads object counter
object_counter = ObjectCounter()

#Vars
spiritlevel = 0 #Random number to determine which sound to play aka your "sprirt"
people = 0 #Counter for the number of people detected
frameholder =0 #Counts frames which is used for detection in change of # of people.



# Load the YOLO model
model = YOLO(r"C:\Users\deany\yolo11x.pt")

# Open the webcam
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)

    # Puts the boxes around the people found in frame of the webcam
    for result in results:
        for box in result.boxes:
        
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            confidence = box.conf[0]  # Confidence score
            class_id = int(box.cls[0])  # Class ID

            
            if class_id == 0 and confidence > .6 :  # If it is a "person" with over 60% confidence 
                
                
                if frameholder == 0:
                    frameholder = 1
                elif frameholder == 1:
                    frameholder = 2
                elif frameholder == 2:
                    frameholder = 3
                elif frameholder == 3:
                    frameholder = 4
                elif frameholder == 4:
                    frameholder = 5
                elif frameholder == 5:
                    
 
              
#                    if object_counter.count_objects(frame, 0, frame, 0) > people:
                    spiritlevel = random.randint(1, 10) 
                        
                    if spiritlevel > 5:
                        playsound(r"C:\Users\deany\OneDrive\Documents\Python\SY3-16\quothello-therequot-158832.mp3")
                        time.sleep(10)
                    elif spiritlevel < 5:
                        playsound(r"C:\Users\deany\OneDrive\Documents\Python\SY3-16\frantic-screaming-213549.mp3")
                        time.sleep(10) 
 


                    #people = object_counter.count_objects(frame)
                    frameholder = 0    
                    


                          
                #print ("Human detected : Line 51") #Marker 



                # Draws the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) #Red box surrounds object
                
                label = f"Person: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


    # Display webcam window
    cv2.imshow("Webcam", frame)

    # Hold Q to kill the program!  (PRESSING X WILL NOT WORK!!!!!!!!!!!!!!!!!!!!!)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Closes the webcam window
cap.release()
