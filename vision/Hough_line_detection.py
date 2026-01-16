#!/usr/bin/env python3
import cv2 as cv
import numpy as np

# Učitavanje i postavljanje slike na format 640x640
img = cv.imread("/home/tomo/Faks/ProjektE/opencv_tutorial/Photos/dog.jpg")
img = cv.resize(img, (640, 640))
cv.imshow("Original", img)

# Stvaranje prazne slike
blank = np.zeros_like(img)

# Pretvaranje u GrayScale i primjena CLAHE za izoštravanje
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)

# Primjena Canny filtra
canny = cv.Canny(gray, 100, 200)
cv.imshow("Canny", canny)

# Primjena Hough detektora za slike
def detect_lines(canny_img, min_len=100, max_gap=4):
    return cv.HoughLinesP(canny_img, 1, np.pi/180, threshold=30,
                          minLineLength=min_len, maxLineGap=max_gap)

# Crtanje linija 
def draw_lines(base_img, lines, color=(0,0,255), thickness=2):
    img_copy = base_img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(img_copy, (x1, y1), (x2, y2), color, thickness)
    else:
        print("No lines detected!")
    return img_copy

# Prva detekcija
lines = detect_lines(canny)
line_img = draw_lines(blank, lines)
cv.imshow("Detected Wires - Pass 1", line_img)

# Iteriranje preko Canny slika
def iterative_hough(canny_img, iterations=3, min_len=200, max_gap=5):
    result = canny_img.copy()
    for i in range(iterations):
        lines = detect_lines(result, min_len, max_gap)
        temp = np.zeros_like(result)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv.line(temp, (x1, y1), (x2, y2), 255, 1)
            result = cv.bitwise_or(result, temp)
            print(f"Iteration {i+1}: {len(lines)} lines reinforced")
        else:
            print(f"Iteration {i+1}: No lines found")
    return result

# Iteriranje
refined_edges = iterative_hough(canny, iterations=10)
cv.imshow("Iterative Hough Refinement", refined_edges)

# Završna verzija slike nakon iteriranja
final_lines = detect_lines(refined_edges)
final_img = draw_lines(img, final_lines)
cv.imshow("Final Detected Wires", final_img)

cv.waitKey(0)
cv.destroyAllWindows()
