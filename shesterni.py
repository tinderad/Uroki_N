import cv2 as cv
import parameters as ps


def recognition(img):
    font = cv.QT_FONT_NORMAL
    frame = cv.resize(img, (800, 600), interpolation=cv.INTER_AREA)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    hsv = cv.blur(hsv, (5, 5))
    mask = cv.inRange(hsv, ps.mask_test[0], ps.mask_test[1])
    contours_info = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours = contours_info[0]
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    (x, y, w, h) = cv.boundingRect(contours[0])
    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    tan = w / h
    if tan > ps.approx_tangent:
        cv.putText(frame, 'Shesternia-1', (x, y - 5), font, 2, (0, 0, 0), 2, cv.LINE_AA)
    else:
        cv.putText(frame, 'Shesternia-2', (x, y - 5), font, 2, (0, 0, 0), 2, cv.LINE_AA)
    return frame


def get_frame(path):
    img = cv.imread(path)
    frame = cv.resize(img, (800, 600), interpolation=cv.INTER_AREA)
    return frame


img1 = get_frame("assets/1.jpg")
img2 = get_frame("assets/2.jpg")

cv.imshow("1", img1)
cv.imshow("2", img2)
cv.waitKey(2000)
cv.imshow("1", recognition(img1))
cv.imshow("2", recognition(img2))

while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()
