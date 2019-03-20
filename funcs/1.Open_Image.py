import cv2 as cv

img = cv.imread('2.JPG') 
frame = cv.resize(img, (800,600), interpolation = cv.INTER_AREA)
cv.imshow("Differential",frame)
                         
while(True):
    if cv.waitKey() == 27:
        break     
    
cv.destroyAllWindows()