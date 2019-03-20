import cv2 as cv
import numpy as np
img = cv.imread('1.JPG') #,cv.IMREAD_GRAYSCALE)
frame = cv.resize(img, (800,600), interpolation = cv.INTER_AREA)
hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)           
hsv = cv.blur(hsv,(5,5))             
mask = cv.inRange(hsv,(159,169,182),(187,255,255))                      

contours_info = cv.findContours(mask,cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
contours = contours_info[1]
    
if contours:
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    cv.drawContours(frame,contours,0,(255,255,0),3)
    (x,y,w,h) = cv.boundingRect(contours[0])    
    cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    
cv.imshow("Contours",frame)
                         
while(True):
    if cv.waitKey() == 27:
        break                         
cv.destroyAllWindows()