# Import Neccessary libraries
import cv2
import numpy as np

# Create Video Capture Instance
camera = cv2.VideoCapture("cars.mp4")
# Create an array for ellipse of 9x4 dimension
es = cv2.getStructuringElement( cv2.MORPH_ELLIPSE, (9,4))
# Create a 5x5 kernel of ones
kernel = np.ones((5,5),np.uint8)
background = None
# Infinite loop unless enter is pressed
while (True):
    ret, frame = camera.read()
    if background is None:
        background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        background = cv2.GaussianBlur(background, (21, 21), 0)
        continue
    # Convert into grayscale and apply GaussianBlur.
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
    # Apply a threshold, so as to obtain a black and white image, and dilate the image so holes and imperfections get normalized.
    diff = cv2.absdiff(background, gray_frame)
    diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    diff = cv2.dilate(diff, es, iterations = 2)
    # Find contours
    image, cnts, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Loop though countours
    for c in cnts:
        if cv2.contourArea(c) < 1500:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Show Windowss
    cv2.imshow("contours", frame)
    cv2.imshow("dif", diff)
    # Break loop when Enter is pressed
    if cv2.waitKey(13):
        break

# Destroy windows and remove camera Instance
cv2.destroyAllWindows()
camera.release()