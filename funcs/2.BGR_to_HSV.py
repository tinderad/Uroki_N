import cv2 as cv

img = cv.imread('2.JPG') 
frame = cv.resize(img, (800,600), interpolation = cv.INTER_AREA)

hsv_frame = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
cv.imshow("Differential",frame)
cv.imshow("HSV",hsv_frame)
                         
while(True):
    if cv.waitKey() == 27:
        break     
    
cv.destroyAllWindows()