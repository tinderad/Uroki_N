import cv2 as cv
import numpy as np

img = cv.imread('funcs/2.jpg')  # ,cv.IMREAD_GRAYSCALE)
frame = cv.resize(img, (800, 600), interpolation=cv.INTER_AREA)
cv.imshow("", frame)
cv.waitKey(1000)
hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
hsv = cv.blur(hsv, (5, 5))
mask = cv.inRange(hsv, (6, 7, 6), (100, 95, 96))
cv.imshow("", mask)
cv.waitKey(1000)

contours_info = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
contours = contours_info[0]

def get_error(mask):
    contours_info = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours = contours_info[0]
    if contours:
        contours = sorted(contours, key=cv.contourArea, reverse=True)
        contour = contours[0]
        epsilon = 0.01 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)
        right = approx[np.argmax(approx,axis=0)[0][0]][0][0]
        bot = approx[np.argmax(approx,axis=0)[0][1]][0][1]
        left = approx[np.argmin(approx, axis=0)[0][0]][0][0]
        top = approx[np.argmin(approx, axis=0)[0][1]][0][1]


if contours:
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    contour = contours[0]

    cv.drawContours(frame, contours, 0, (255, 255, 0), 3)
    cv.imshow("", frame)
    cv.waitKey(1000)

    (x, y, w, h) = cv.boundingRect(contour)
    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    epsilon = 0.01 * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon, True)
    #approx = np.squeeze(approx, axis=1)
    #print(approx)
    get_error(mask)
    cv.drawContours(frame, approx, -1, (255, 0, 0), 8)
    cv.imshow("", frame)


while (True):
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()
