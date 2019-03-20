import cv2 as cv

img = cv.imread('2.JPG') 
frame = cv.resize(img, (800,600), interpolation = cv.INTER_AREA)
blur_image = cv.blur(frame,(5,5)) 

cv.imshow("Blur",blur_image)

hsv_frame = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
cv.imshow("Differential",frame)


#mask = cv.inRange(hsv_frame,(150,0,0),(190,255,255))
#cv.imshow("Mask of HSV",mask)
                         
while(True):
    if cv.waitKey() == 27: 
        break     
    
cv.destroyAllWindows()