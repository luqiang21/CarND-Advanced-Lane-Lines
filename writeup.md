##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./output_images/undistorted.png "Road Transformed"
[image3]: ./output_images/binary_combo_example.png "Binary Example"
[image4]: ./output_images/warped_straight_lines.png "Warp Example"
[image5]: ./output_images/color_fit_lines.jpg "Fit Visual"
[image6]: ./output_images/example_output.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

I first imported necessary python packages. Then the calibration parameters (mtx, dst) were computed based on the calibration images provided by Udacity. An image now can be undistorted using the parameters. The image can also be unwarped by selecting four points that should be a rectangle in real world.

A histogram of the bottom half of the unwarped image was taken to find the peaks which corresponding to lanes, thus it is easy to get the lane center and the vehicle center is simply the center of the image. I computed the lane curvatures for each lane. Then I draw lane area on the undistorted image.

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the 5th cell of Advanced_Lane_Lines.ipynb.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using my function cal_undistort(img, objpoints, imgpoints) which includes the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
The parameters mtx and dist of distortion for the camera has been computed from last step. Thus I can just apply the camera parameters to other image like the above one.
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (the 3rd cell in the notebook).  Here's an example of my output for this step.

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called unwarp(img, mtx, dist), which appears the 4th cell in the notebook).  The function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points. The source four points are selected on the image by the file 'pickPoints.py' which gives coordinates when you click within the image.

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 541, 490      | 404, 490       |
| 747, 490      | 899, 490      |
| 254, 681     | 404, 681      |
| 1049, 681      |899, 681       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear nearly parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this (cells 6 and 10 in the notebook):

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in cell 26 within the function process(img, Minv, left_fit, right_fit) and cell 9 in the notebook.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in cell 11 in the notebook.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The hardest part for me is the average over several images which I didn't make it. The weak part of my current implementation is the lane curvature computation. To make it robust, I need to know the implementation of class in python and then make the obtained lane lines stable. Thus, by using class, vehicle's position and lane lines positions, lane polygons could be averaged to get more stable results. Also, different conditions of the road images like brightness, curvature and so on will include noise on the detection of the lanes. More thresholding on more color channels may bring robustness.

To have a stable plotting of lane area, I tried the method from my reviewer to reject weird polygons. It seems improved the performance a bit. More work should be done on the extraction of lane lines with averaging.
