# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"
[image2]: ./examples/lanesLines_thirdPass.jpg "Lane Lines Extrapolated"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

#### Pipeline
My pipeline is defined in `def detect_lane_pipeline`. here I have followed the following steps:
1. Converting the image to Grayscale using the `grayscale()` function.
2. The grayscaled image is then blurred using the `gaussian_blur()` function with a kernel size of 3.
3. Then the image is passed through the Canny Edge Detection `canny()` function with a `low_threshold = 80` and `high_threshold = 160`.
4. To define the ROI, I approxiamated the width and height to the image's half width and height approxiamtely. 
5. Then applied Hough Transaform with `min_line_length = 25` and `max_line_gap = 25`.

#### Draw Line
I modified the `draw_lines()` function using the hint given, in the following way: 
1. Initializing the max and min Y-coordinates and list of X, Y coordinates, slopes and intercepts. 
2. Calculating slope at every step of the loop and appending the respective Y, Y cordinates, slope and intercept.
3. If slope > 0 then it belongs to right line and if slope < 0 then the points belong to left line.
4. Then I find the average intercept, slope, X, Y coordinate and find the min and max X coordinates.


If you'd like to include images to show how the pipeline works, here is how to include an image: 

![image3]: ./test_images/solidYellowCurveDetected.jpg

![image4]: ./test/solidWhiteRightLaneDetected.png


### 2. Shortcomings with your current pipeline

The current short comming with the current pipeline is that it is unable to detect LaneLines which curve out i.e when there is a turning the Lane Lines curve and then the current pipeline is that able to detect as it falls out of ROI.

Another shortcoming could be the speed at which this is processed and detected it need to be more fast.


### 3. Suggest possible improvements to your pipeline

Improvemnt for the current pipeline includes more rhobust Lane Detection using advanced machine learning and image processing and tracking multi-objects on the road to make sure and know where the vehicle is moving.