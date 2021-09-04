import numpy as np
import cv2
import matplotlib.pyplot as plt

#Read original image
image = cv2.imread("project.PNG")
cv2.imshow('Input image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# convert to gray scale image
gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray.png', gray)

# Apply median filter for smoothning
blur_M = cv2.medianBlur(gray, 5)
cv2.imwrite('blur_M.png', blur_M)

# Apply gaussian filter for smoothning
blur_G = cv2.GaussianBlur(gray, (9, 9), 0)
cv2.imwrite('blur_G.png', blur_G)

#Display
blurHorti = np.concatenate((blur_M,blur_G), axis=1)
cv2.imshow('Smoothning Results- Median Blurring and Gaussian Blurring', blurHorti)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Edge Detection using Canny Edge Detector
edge = cv2.Canny(gray, 100, 200)
edge_G = cv2.Canny(blur_G, 100, 200)
cv2.imwrite('edge_G.png', edge_G)
edge_M = cv2.Canny(blur_M, 100, 200)

#Display
edgeHorti = np.concatenate((edge,edge_G,edge_M), axis=1)
cv2.imshow('Canny edge detection Results- on Gray image,Gaussian and Mean Smoothened images', edgeHorti)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Read enhanced image 
img = cv2.imread('edge_G.png', 0)
kernel = np.ones((5, 5), np.uint8)

# Morphological Operations
dilation = cv2.dilate(img, kernel, iterations = 1)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# Adaptive thresholding using Mean and Gaussian filter
thresh1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imwrite('thresh2.png',thresh2)

# Otsu's thresholding
ret3, thresh3 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#Display
threshHorti = np.concatenate((thresh1,thresh2,thresh3), axis=1)
cv2.imshow("Adaptive thresholding using Mean and Gaussian filter and Otsu's Thresholding Results", threshHorti)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Initializing list
Cell_count, x_count, y_count = [], [], []

# Hough Transform with modified Circular parameters
display = cv2.imread("thresh2.png",0)
circles = cv2.HoughCircles(display, cv2.HOUGH_GRADIENT, 1.2, 20, param1 = 50, param2 = 28, 
                           minRadius = 1, maxRadius = 25)

# Circle Detection and labeling using Hough Transformation  
if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
                cv2.circle(display, (x, y), r, (0, 0, 0), 2)
                cv2.rectangle(display, (x - 2, y - 2), (x + 2, y + 2), (0, 0, 0), -1)
                Cell_count.append(r)
                x_count.append(x)
                y_count.append(y)
      
        cir = thresh2-display
        img = cv2.add(cir,gray)
        cv2.waitKey(0)
  
# Display the count of White Blood Cells
Horti1 = np.concatenate((gray,blur_G,edge_G,dilation), axis=1)
Horti2= np.concatenate((closing,thresh2, cir, img), axis=1)


cv2.imshow('Gray_Img ,Blurred_Img ,Canny_edge,Dilation', Horti1)
cv2.imshow('Closing, Thresholding, Detection, Counting', Horti2)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('number of cells =' ,len(Cell_count))
print('Radius of cells =', Cell_count) 
print('x-axis =', x_count)     
print('y-axis =', y_count)
