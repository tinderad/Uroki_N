import cv2 as cv
from random import randint
import numpy as np
import parameters as ps
from tqdm import tqdm

img1 = "funcs/1.jpg"
img2 = "funcs/2.jpg"
epsilon = int(input("epsilon: "))
step = int(input("step: "))
masks = np.array([[0, 0, 0], [0, 0, 0]])
print("Creating masks map... \n")
for h in tqdm(range(ps.h - epsilon, ps.h + epsilon, step)):
    if ps.h - epsilon < 0: pass
    for s in range(ps.s - epsilon, ps.s + epsilon, step):
        if ps.s - epsilon < 0: pass
        for v in range(ps.v - epsilon, ps.v + epsilon, step):
            if ps.v - epsilon < 0: pass
            for H in range(ps.H - epsilon, ps.H + epsilon, step):
                if ps.H - epsilon < 0: pass
                for S in range(ps.S - epsilon, ps.S + epsilon, step):
                    if ps.S - epsilon < 0: pass
                    for V in range(ps.V - epsilon, ps.V + epsilon, step):
                        if ps.V - epsilon < 0: pass
                        masks = np.concatenate((masks, [[h, s, v], [H, S, V]]), axis=0)

print("Done, masks created: " + str(len(masks)))


# frame_hsv1 = cv.cvtColor(frame1,cv.COLOR_BGR2HSV)
# frame_hsv2 = cv.cvtColor(frame2,cv.COLOR_BGR2HSV)


def get_frame(path):
    img = cv.imread(path)
    frame = cv.resize(img, (800, 600), interpolation=cv.INTER_AREA)
    return frame


def set_bounds(path):
    while (True):
        newframe = get_frame(path)
        x = cv.getTrackbarPos("x", "Sliders")
        x1 = cv.getTrackbarPos("x1", "Sliders")
        y = cv.getTrackbarPos("y", "Sliders")
        y1 = cv.getTrackbarPos("y1", "Sliders")
        cv.rectangle(newframe, (x, y), (x1, y1), (randint(0, 255), randint(0, 255), randint(0, 255)), 2)
        cv.imshow('1', newframe)
        if cv.waitKey(1) == 27:
            break
    return x, x1, y, y1


def __get_error(mask, roi):
    contours_info = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours = contours_info[0]
    if contours:
        contours = sorted(contours, key=cv.contourArea, reverse=True)
        contour = contours[0]
        epsilon = 0.01 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)
        right = approx[np.argmax(approx, axis=0)[0][0]][0][0]
        bot = approx[np.argmax(approx, axis=0)[0][1]][0][1]
        left = approx[np.argmin(approx, axis=0)[0][0]][0][0]
        top = approx[np.argmin(approx, axis=0)[0][1]][0][1]
        x, x1, y, y1 = roi
        error = (abs(left - x) + abs(right - x1) + abs(top - y) + abs(bot - y1)) / 4
        return error
    else:
        return -1


def __get_total_error(img1, img2, roi1, roi2, HSV):
    hsv1 = cv.cvtColor(img1, cv.COLOR_BGR2HSV)
    hsv1 = cv.blur(hsv1, (5, 5))
    mask1 = cv.inRange(hsv1, HSV)
    hsv2 = cv.cvtColor(img2, cv.COLOR_BGR2HSV)
    hsv2 = cv.blur(hsv2, (5, 5))
    mask2 = cv.inRange(hsv2, HSV)
    error1 = __get_error(mask1, roi1)
    error2 = __get_error(mask2, roi2)
    if error1 is not -1 and error2 is not -1:
        total_error = (error1 + error2) / 2
        return total_error
    else: return -1

def empty(a):
    pass


def error(HSV):
    global roi1
    global roi2
    global img1
    global img2
    return __get_total_error(get_frame(img1),get_frame(img2),roi1,roi2,HSV)


cv.namedWindow("Sliders")
cv.createTrackbar("x", "Sliders", 0, 800, empty)
cv.createTrackbar("x1", "Sliders", 0, 800, empty)
cv.createTrackbar("y", "Sliders", 0, 600, empty)
cv.createTrackbar("y1", "Sliders", 0, 600, empty)

roi1 = set_bounds(img1)
roi2 = set_bounds(img2)

get_error_label = np.vectorize(error)
error_label = get_error_label(masks)
#тут надо argmin взять от слоя ошибок и выявить лучший HSV. С первого раза вряд ли запустится.
cv.destroyAllWindows()
