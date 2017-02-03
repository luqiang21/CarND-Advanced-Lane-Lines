#First, I'll compute the camera calibration using chessboard images
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for fname in images:
	img =  mpimg.imread(fname)
	gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

	# Find the chessboard corners
	ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

	# If found, add object points, image points
	if ret == True:
		objpoints.append(objp)
		imgpoints.append(corners)

		# Draw and display the corners
		img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
		cv2.imshow('img',img)
		cv2.waitKey(50)
cv2.destroyAllWindows()

# Next, read in a image and undistort it.
# Read in an image
img = mpimg.imread('camera_cal/calibration1.jpg')

# a function that takes an image, object points, and image points
# performs the camera calibration, image distortion correction and 
# returns the undistorted image
def cal_undistort(img, objpoints, imgpoints):
	# Use cv2.calibrateCamera and cv2.undistort()
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[0:2],None,None)
	undist = cv2.undistort(img, mtx, dist, None, mtx)
	return undist, mtx, dist

undistorted, mtx, dist = cal_undistort(img, objpoints, imgpoints)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undistorted)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

# Next, perform thresholding, output binary image


image = mpimg.imread('test_images/straight_lines1.jpg')
img = image


undistorted, mtx, dist = cal_undistort(img, objpoints, imgpoints)


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undistorted)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


# Read in an image

# MODIFY THIS FUNCTION TO GENERATE OUTPUT 
# THAT LOOKS LIKE THE IMAGE ABOVE
def unwarp(img, mtx, dist):
	# Pass in your image into this function
	# Write code to do the following steps
	# 1) Undistort using mtx and dist
	# 2) Convert to grayscale
	# 3) Find the chessboard corners
	# 4) If corners found: 
			# a) draw corners
			# b) define 4 source points src = np.float32([[,],[,],[,],[,]])
				 #Note: you could pick any four of the detected corners 
				 # as long as those four corners define a rectangle
				 #One especially smart way to do this would be to use four well-chosen
				 # corners that were automatically detected during the undistortion steps
				 #We recommend using the automatic detection of corners in your code
			# c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
			# d) use cv2.getPerspectiveTransform() to get M, the transform matrix
			# e) use cv2.warpPerspective() to warp your image to a top-down view

	undist = cv2.undistort(img, mtx, dist, None, mtx)
	gray = cv2.cvtColor(undist,cv2.COLOR_RGB2GRAY)
	img = gray
	img_size = (img.shape[1],img.shape[0])

	#obtained through pickPoints.py
	ul = [541, 490]   # upper left point
	ur = [747, 490] # upper right point
	ll = [254, 681] # lower right point
	lr = [1049, 681] # lower left point

	ul = [585, 451]   # upper left point
	ur = [688, 451] # upper right point
	ll = [249, 681] # lower right point
	lr = [1049, 681] # lower left point

	src = np.float32([ul, ur, lr, ll])
	offset = 100
	ul_new = [254 + 100, 490]
	ur_new = [1049 - 100, 490]
	ll_new = [254 + 100, 681]
	lr_new = [1049 - 100, 681]

	ul_new = [249 + 150, 451]   # upper left point
	ur_new = [1049 - 150, 451] # upper right point
	ll_new = [249 + 150, 681] # lower right point
	lr_new = [1049 - 150, 681] # lower left point
	dst = np.float32([[offset, offset], [img_size[0] - offset, offset],
		[img_size[0] - offset, img_size[1] - offset], [offset, img_size[1] - offset]])

	dst = np.float32([ul_new, ur_new, lr_new, ll_new])
	plt.imshow(img)

	M = cv2.getPerspectiveTransform(src, dst)
	img_size = (img.shape[1],img.shape[0])
	warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)

	return warped, M

top_down, perspective_M = unwarp(img, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
img = cv2.undistort(img, mtx, dist, None, mtx)
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)




# Edit this function to create your own pipeline.
def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
	img = np.copy(img)
	# Convert to HSV color space and separate the V channel
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
	l_channel = hls[:,:,1]
	s_channel = hls[:,:,2]
	
	hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
	v_channel = hsv[:,:,2]
	s_channel = hsv[:,:,1]

	# Sobel x
	sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
	abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
	scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
	
	# Threshold x gradient
	sxbinary = np.zeros_like(scaled_sobel)
	sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
	
	# Threshold color channel
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
	# Stack each channel
	# Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
	# be beneficial to replace this channel with something else.
	color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
	# Combine the two binary thresholds
	combined_binary = np.zeros_like(sxbinary)
	combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
	
	return color_binary, combined_binary
	

result = pipeline(image)[1]


# # Plot the result
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
#
# ax1.imshow(image)
# ax1.set_title('Original Image', fontsize=40)
#
# ax2.imshow(result, cmap='gray')
# ax2.set_title('Pipeline Result', fontsize=40)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


plt.show()