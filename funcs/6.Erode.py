import cv2 as cv

img = cv.imread('2.JPG') 
frame = cv.resize(img, (640,480), interpolation = cv.INTER_AREA)

hsv_frame = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
#cv.imshow("Differential",frame)

mask = cv.inRange(hsv_frame,(150,0,0),(190,255,255))
cv.imshow("Mask of HSV",mask)

mask_Erode = cv.erode(mask,None,iterations=2)
mask_Dilate = cv.dilate(mask_Erode,None,iterations=2)
cv.imshow("Erode",mask_Erode)
cv.imshow("Erode and Dilate",mask_Dilate)
                         
while(True):
    if cv.waitKey(0) == 27: 
        break     
    
cv.destroyAllWindows()