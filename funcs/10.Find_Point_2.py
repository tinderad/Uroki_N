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
    contour = contours[0]
    (x,y,w,h) = cv.boundingRect(contour)    
    epsilon = 0.01*cv.arcLength(contour,True)
    approx = cv.approxPolyDP(contour,epsilon,True)
    cv.drawContours(frame, approx, -1, (0, 255, 0), 10)
    approx = np.squeeze(approx, axis=1)
    x0 = x + int(w * 0.8) 
    y0 = y + int(h * 0.9)
    x1 = x + int(w*0.9)
    y1 = y + int(h*1.1)
    flag = False
    for point in approx:
        X,Y = point
        if x0 < X < x1 and y0 < Y < y1:
            flag = True
            break    
    print(flag)
cv.imshow("Contours",frame)
                         
while(True):
    if cv.waitKey() == 27:
        break                         
cv.destroyAllWindows()