import cv2 as cv

img = cv.imread('1.jpg')
frame = cv.resize(img, (800,600), interpolation = cv.INTER_AREA)
frame_hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)

def empty(a):
    pass

cv.namedWindow("Sliders")
cv.createTrackbar("h","Sliders",0,255,empty)
cv.createTrackbar("H","Sliders",0,255,empty)
cv.createTrackbar("s","Sliders",0,255,empty)
cv.createTrackbar("S","Sliders",0,255,empty)
cv.createTrackbar("v","Sliders",0,255,empty)
cv.createTrackbar("V","Sliders",0,255,empty)

while(True):
    h = cv.getTrackbarPos("h","Sliders")
    H = cv.getTrackbarPos("H","Sliders")
    s = cv.getTrackbarPos("s","Sliders")
    S = cv.getTrackbarPos("S","Sliders")
    v = cv.getTrackbarPos("v","Sliders")
    V = cv.getTrackbarPos("V","Sliders")

    mask = cv.inRange(frame_hsv,(h,s,v),(H,S,V))
    cv.imshow('Mask', mask)                    

    result = cv.bitwise_and(frame,frame, mask = mask)                       
    cv.imshow('And', result)                          
    if cv.waitKey(1) == 27:
        break

cv.destroyAllWindows()