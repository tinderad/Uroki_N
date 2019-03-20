import cv2 as cv

img = cv.imread('2.JPG') 
frame = cv.resize(img, (640,480), interpolation = cv.INTER_AREA)

hsv_frame = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
cv.imshow("Differential",frame)
cv.imshow("HSV",hsv_frame)

mask_of_frame = cv.inRange(frame,(0,0,0),(200,255,255))
cv.imshow("Mask of Differential",mask_of_frame)

mask_of_hsv = cv.inRange(hsv_frame,(150,0,0),(190,255,255))
cv.imshow("Mask of HSV",mask_of_hsv)
                         
while(True):
    if cv.waitKey(0) == 27: 
        break     
    
cv.destroyAllWindows()