import cv2 as cv
from random import randint
import numpy as np
import parameters as ps
from tqdm import tqdm


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
        error = ((abs(left - x) / abs(x - x1)) + (abs(right - x1) / abs(x - x1)) + (abs(top - y) / abs(y - y1)) + (abs(
            bot - y1) / abs(y - y1)))
        return error
    else:
        return -1


def __get_total_error(img1, img2, roi1, roi2, HSV):
    HSV = tuple(map(int, HSV[0])), tuple(map(int, HSV[1]))
    hsv1 = cv.cvtColor(img1, cv.COLOR_BGR2HSV)
    hsv1 = cv.blur(hsv1, (5, 5))
    mask1 = cv.inRange(hsv1, HSV[0], HSV[1])
    hsv2 = cv.cvtColor(img2, cv.COLOR_BGR2HSV)
    hsv2 = cv.blur(hsv2, (5, 5))
    mask2 = cv.inRange(hsv2, HSV[0], HSV[1])
    error1 = __get_error(mask1, roi1)
    error2 = __get_error(mask2, roi2)
    if error1 is not -1 and error2 is not -1:
        total_error = (error1 + error2)
        return total_error
    else:
        return -1


def get_tg(HSV, path):
    img = cv.imread(path)
    frame = cv.resize(img, (800, 600), interpolation=cv.INTER_AREA)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    hsv = cv.blur(hsv, (5, 5))
    mask = cv.inRange(hsv, HSV[0], HSV[1])
    contours_info = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours = contours_info[0]
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    contour = contours[0]
    (x, y, w, h) = cv.boundingRect(contour)
    tg = abs(w) / abs(h)
    return tg


def empty(a):
    pass


def error(HSV):
    global roi1
    global roi2
    global img1
    global img2
    return __get_total_error(get_frame(img1), get_frame(img2), roi1, roi2, (tuple(HSV[0]), tuple(HSV[1])))


img1 = "assets/1.jpg"
img2 = "assets/2.jpg"
epsilon = int(input("epsilon: "))
step = int(input("step: "))
skip = int(input("skip: "))
masks = np.array([[[0, 0, 0], [0, 0, 0]]])
print("Creating masks map... \n")
for h in tqdm(range(ps.h - epsilon, ps.h + epsilon + 1, step)):
    if h > 0:
        for s in range(ps.s - epsilon, ps.s + epsilon + 1, step):
            if s > 0:
                for v in range(ps.v - epsilon, ps.v + epsilon + 1, step):
                    if v > 0:
                        for H in range(ps.H - epsilon, ps.H + epsilon + 1, step):
                            if H > 0:
                                for S in range(ps.S - epsilon, ps.S + epsilon + 1, step):
                                    if S > 0:
                                        for V in range(ps.V - epsilon, ps.V + epsilon + 1, step):
                                            if V > 0:
                                                masks = np.concatenate((masks, [[[h, s, v], [H, S, V]]]), axis=0)

masks = masks[1:]
print("Done, masks created: " + str(len(masks)) + "\n")

cv.namedWindow("Sliders")
cv.createTrackbar("x", "Sliders", 0, 800, empty)
cv.createTrackbar("x1", "Sliders", 0, 800, empty)
cv.createTrackbar("y", "Sliders", 0, 600, empty)
cv.createTrackbar("y1", "Sliders", 0, 600, empty)

roi1 = set_bounds(img1)
roi2 = set_bounds(img2)

<<<<<<< HEAD
print("Current mask error: " + str(round(error(ps.mask_HSV) * 100, 2)) + "%")
=======
# get_error_label = np.vectorize(error)
# error_label = get_error_label(masks)
print("Current mask error:" + str(error(ps.mask_HSV) * 100) + "%")
>>>>>>> 0f73ace340f0f94ccdc4a9c6e395d32cc291168f
print("Starting calculation... \n")
error_label = np.array([])
for mask in tqdm(masks[:60:skip]):
    error_label = np.append(error_label, error(mask))
NewOptimalMask = masks[np.argmin(error_label)]
print("New optimal mask: ")
print(NewOptimalMask)
<<<<<<< HEAD
print("error: " + str(round(error(NewOptimalMask) * 100, 2)) + "%")
print(
    "New optimal tangent:  " + str((get_tg(NewOptimalMask, img1) + get_tg(NewOptimalMask, img2)) / 2))

=======
print("error: " + str(error(tuple(NewOptimalMask)) * 100) + "%")
# тут надо argmin взять от слоя ошибок и выявить лучший HSV. С первого раза вряд ли запустится.
>>>>>>> 0f73ace340f0f94ccdc4a9c6e395d32cc291168f
cv.destroyAllWindows()
