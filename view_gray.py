#!/usr/bin/python

import cv2

# some vars
# read /dev/video0
cap = cv2.VideoCapture(0)

# main loop
while True:
    ret, img = cap.read()

    # skip if no image
    if img is None:
        continue

    # convert image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # display img
    cv2.imshow('image', img)

    # end of main loop if 'esc' press
    key = cv2.waitKey(5) & 0xFF
    if key == 27:
        break

cv2.destroyAllWindows()
cv2.VideoCapture(0).release()
