#!/usr/bin/python2

# decode 7 segments digit value in image "lcd.jpg"
# test on debian stretch with opencv v2.4.9.1 (from dist package python-opencv)

import cv2
import matplotlib.pyplot as plt
import numpy as np

# some const
DIGIT_VALUE = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 1, 0): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9
}

# some vars
l_digits = list()

# read image
img = cv2.imread('simple_lcd_7seg.jpg')

# convert image (img -> gray -> blur)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

# detect digit(s): threshold -> erode
(ret, img_thresh) = cv2.threshold(img_blur, 50, 100, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
kernel = np.ones((10, 10), np.uint8)
img_thresh_erode = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
# digit(s) contour detect
l_dig_cnt, _ = cv2.findContours(img_thresh_erode.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
l_sel_dig_cnt = list()
# for every contour: convert contour point list to box and select specific size
for dig_cnt in l_dig_cnt:
    (x, y, w, h) = cv2.boundingRect(dig_cnt)
    if w >= 20 and 50 <= h <= 300:
        l_sel_dig_cnt.append(dig_cnt)
        # draw box around digit on img_cor
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)

# search segment size
for sel_dig_cnt in l_sel_dig_cnt:
    (x, y, w, h) = cv2.boundingRect(sel_dig_cnt)
    img_digit = img_thresh[y:y + h, x:x + w]
    (rh, rw) = img_digit.shape
    (dw, dh) = (int(rw * 0.25), int(rh * 0.15))
    dhc = int(rh * 0.05)
    # define the 7 segments box: ((x, y pos), (w, h box size))
    segments = [
        ((0, 0), (w, dh)),
        ((0, 0), (dw, h // 2)),
        ((w - dw, 0), (w, h // 2)),
        ((0, (h // 2) - dhc), (w, (h // 2) + dhc)),
        ((0, h // 2), (dw, h)),
        ((w - dw, h // 2), (w, h)),
        ((0, h - dh), (w, h))
    ]
    # init seg status list : 0 is off (default), 1 is on
    seg_status = [0] * len(segments)
    # populate seg status with on segment
    for (sel_index, ((xa, ya), (xb, yb))) in enumerate(segments):
        seg_roi = img_digit[ya:yb, xa:xb]
        area = (xb - xa) * (yb - ya)
        sum_pix = cv2.countNonZero(seg_roi)
        # 30% of segment must be full
        if sum_pix / float(area) > 0.3:
            seg_status[sel_index] = 1
    # digit identification
    try:
        l_digits.append(DIGIT_VALUE[tuple(seg_status)])
    except KeyError:
        print('digit not found (segments %s is unknown)' % seg_status)
# digit(s) list in read order
l_digits.reverse()
# display result
print('find digit(s) %s' % l_digits)

# display all img for tuning purpose
fig = plt.figure()
fig.add_subplot(2, 2, 1).imshow(img)
plt.gca().set_title('img')
fig.add_subplot(2, 2, 3).imshow(img_thresh, cmap='gray')
plt.gca().set_title('img_thresh')
fig.add_subplot(2, 2, 4).imshow(img_thresh_erode, cmap='gray')
plt.gca().set_title('img_thresh_erode')
plt.show()
