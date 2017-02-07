#First, I'll compute the camera calibration using chessboard images
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#### obtain calibration parameters
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

# a function that takes an image, object points, and image points
# performs the camera calibration, image distortion correction and 
# returns the undistorted image
def cal_undistort(img, objpoints, imgpoints):
	# Use cv2.calibrateCamera and cv2.undistort()
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[0:2],None,None)
	undist = cv2.undistort(img, mtx, dist, None, mtx)
	return undist, mtx, dist


# MODIFY THIS FUNCTION TO GENERATE OUTPUT 
# THAT LOOKS LIKE THE IMAGE ABOVE
def unwarp(img, mtx, dist):
	# Pass in your image into this function
	# Write code to do the following steps


	undist = cv2.undistort(img, mtx, dist, None, mtx)
	# gray = cv2.cvtColor(undist,cv2.COLOR_RGB2GRAY)
	# img = gray
	img_size = (img.shape[1],img.shape[0])

	src = np.float32([ul, ur, lr, ll])
	offset = 100

	dst = np.float32([[offset, offset], [img_size[0] - offset, offset],
		[img_size[0] - offset, img_size[1] - offset], [offset, img_size[1] - offset]])

	dst = np.float32([ul_new, ur_new, lr_new, ll_new])
	plt.imshow(img)

	M = cv2.getPerspectiveTransform(src, dst)
	img_size = (img.shape[1],img.shape[0])
	warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)

	return warped, M


# Edit this function to create your own pipeline.
def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):

	########### Apply a distortion correction to raw images.
	undistorted = cv2.undistort(img, mtx, dist, None, mtx)


	########### Use color transforms, gradients, etc., to create a thresholded binary image.
	# Convert to HSV color space and separate the V channel
	hls = cv2.cvtColor(undistorted, cv2.COLOR_RGB2HLS).astype(np.float)
	l_channel = hls[:,:,1]
	s_channel = hls[:,:,2]
	
	hsv = cv2.cvtColor(undistorted, cv2.COLOR_RGB2HSV).astype(np.float)
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


	########### Apply a perspective transform to rectify binary image ("birds-eye view").
	img_size = (img.shape[1],img.shape[0])
	src = np.float32([ul, ur, lr, ll])
	dst = np.float32([ul_new, ur_new, lr_new, ll_new])
	# plt.imshow(img)

	M = cv2.getPerspectiveTransform(src, dst)
	img_size = (img.shape[1],img.shape[0])
	warped = cv2.warpPerspective(combined_binary, M, img_size, flags=cv2.INTER_LINEAR)

	return color_binary, warped
	

# Next, read in a image and undistort it.
# Read in an image
img = mpimg.imread('camera_cal/calibration1.jpg')
undistorted, mtx, dist = cal_undistort(img, objpoints, imgpoints)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=20)
ax2.imshow(undistorted)
ax2.set_title('Undistorted Image', fontsize=20)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

# Next, perform thresholding, output binary image


img = mpimg.imread('test_images/straight_lines1.jpg')
# img = image


undistorted, mtx, dist = cal_undistort(img, objpoints, imgpoints)


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=20)
ax2.imshow(undistorted)
ax2.set_title('Undistorted Image', fontsize=20)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


#The four points are obtained through pickPoints.py
ul = [541, 490]   # upper left point
ur = [747, 490] # upper right point
ll = [254, 681] # lower left point
lr = [1049, 681] # lower right point

# ul = [544, 481]   # upper left point
# ur = [738, 481] # upper right point
# ll = [205, 714] # lower left point
# lr = [1086, 714] # lower right point
offset = 150

ul_new = [ll[0] + offset, ul[1]]   # upper left point
ur_new = [lr[0] - offset, ur[1]] # upper right point
ll_new = [ll[0] + offset, ll[1]] # lower left point
lr_new = [lr[0] - offset, lr[1]] # lower right point
# print(ul_new, ur_new, ll_new, lr_new)


color=[255, 0, 0]
top_down, perspective_M = unwarp(img, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
img_line = cv2.undistort(img, mtx, dist, None, mtx)
cv2.line(img_line, tuple(ll), tuple(ul), color,10)#
cv2.line(img_line, tuple(lr), tuple(ur), color,10)#
cv2.line(img_line, tuple(ul), tuple(ur), color,10)#
ax1.imshow(img_line)
ax1.set_title('Undistorted Image', fontsize=20)

cv2.line(top_down, tuple(ll_new), tuple([ul_new[0], 0]), color,10)#
cv2.line(top_down, tuple(lr_new), tuple([ur_new[0], 0]), color,10)#
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize=20)

plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
# plt.show()


result = pipeline(img)[1]

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img_line)
ax1.set_title('Original Image', fontsize=20)

ax2.imshow(result, cmap='gray')
ax2.set_title('Pipeline Result', fontsize=20)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


binary_warped = result
# Assuming you have created a warped binary image called "binary_warped"
# Take a histogram of the bottom half of the image
histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
# Create an output image to draw on and  visualize the result
out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
# Find the peak of the left and right halves of the histogram
# These will be the starting point for the left and right lines
midpoint = np.int(histogram.shape[0]/2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint








# Choose the number of sliding windows
nwindows = 9
# Set height of windows
window_height = np.int(binary_warped.shape[0]/nwindows)
# Identify the x and y positions of all nonzero pixels in the image
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
# Current positions to be updated for each window
leftx_current = leftx_base
rightx_current = rightx_base
# Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 50
# Create empty lists to receive left and right lane pixel indices
left_lane_inds = []
right_lane_inds = []

# Step through the windows one by one
for window in range(nwindows):
	# Identify window boundaries in x and y (and right and left)
	win_y_low = binary_warped.shape[0] - (window+1)*window_height
	win_y_high = binary_warped.shape[0] - window*window_height
	win_xleft_low = leftx_current - margin
	win_xleft_high = leftx_current + margin
	win_xright_low = rightx_current - margin
	win_xright_high = rightx_current + margin
	# Draw the windows on the visualization image
	cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
	cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
	# Identify the nonzero pixels in x and y within the window
	good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
	good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
	# Append these indices to the lists
	left_lane_inds.append(good_left_inds)
	right_lane_inds.append(good_right_inds)
	# If you found > minpix pixels, recenter next window on their mean position
	if len(good_left_inds) > minpix:
		leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
	if len(good_right_inds) > minpix:
		rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

# Concatenate the arrays of indices
left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

# Extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds]
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]

# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)



# Generate x and y values for plotting
ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)




# put words on an image
font = cv2.FONT_HERSHEY_SIMPLEX
print('left lane center,', leftx_base, 'right lane center,', rightx_base)
lane_center = (leftx_base + rightx_base) / 2
vehicle_center = result.shape[1] / 2
print('centers, lane,', lane_center, 'vehicle,', vehicle_center)
difference = vehicle_center - lane_center
if difference > 0:
	print('Vehicle is', round(difference * 3.7/700, 2), 'm right of center.')
	cv2.putText(result, ('Vehicle is', round(difference * 3.7/700, 2), 'm right of center.'), (50, 50), font, 1, (255, 255, 255), 2)
elif difference < 0:
	print('Vehicle is', round(-difference * 3.7/700, 2), 'm left of center.')
	cv2.putText(result, 'Vehicle is' + str(round(difference * 3.7/700, 2)) + 'm left of center.', (50, 50), font, 1, (0, 0, 0), 2)
else:
	print('Vehicle is on the center.')



'''for later images'''
# Assume you now have a new warped binary image
# from the next frame of video (also called "binary_warped")
# It's now much easier to find line pixels!
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
margin = 100
left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

# Again, extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds]
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]
# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
# Generate x and y values for plotting
ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


'''visualize result'''
# Create an image to draw on and an image to show the selection window
out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
window_img = np.zeros_like(out_img)
# Color in left and right line pixels
out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

# Generate a polygon to illustrate the search window area
# And recast the x and y points into usable format for cv2.fillPoly()
left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
left_line_pts = np.hstack((left_line_window1, left_line_window2))
right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
right_line_pts = np.hstack((right_line_window1, right_line_window2))

# Draw the lane onto the warped blank image
cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
plt.imshow(result)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)






img = result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()

histogram = np.sum(img[img.shape[0]/2:,:], axis=0)
plt.plot(histogram)
ax1.imshow(histogram)
#
# '''Drawing the lines back down onto the road'''
# # Create an image to draw the lines on
# warp_zero = np.zeros_like(result).astype(np.uint8)
# color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
#
# # Recast the x and y points into usable format for cv2.fillPoly()
# pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
# pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
# pts = np.hstack((pts_left, pts_right))
#
# # Draw the lane onto the warped blank image
# cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
#
# # Warp the blank back to original image space using inverse perspective matrix (Minv)
# newwarp = cv2.warpPerspective(color_warp, perspective_M, (img.shape[1], img.shape[0]))
# # Combine the result with the original image
# result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)
# plt.imshow(result)

plt.show()