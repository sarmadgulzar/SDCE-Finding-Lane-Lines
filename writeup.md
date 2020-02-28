# Pipeline

My lane detection pipeline consists of **5 steps** which are as follows:
* Find **White** and **Yellow** regions of the image
* Apply **Gaussian Blurring** and **Canny Edge Detection**
* Mask everything except our **Region of Interest**
* Apply **Probabilistic Hough Line Transform** to find straight lines
* **Draw** the lines and apply **Linear Regression** to find just **two** (*Left* and *Right*) lines

# Potential Shortcomings
1. One major and obvious shortcoming, as seen in [challenge.mp4](https://youtu.be/be8Es080aOA) video, is that this pipeline does not perform well in **varying lighting conditions**.


2. This pipeline also does not perform well on **curved lines** because we are fitting a straight line through the points.

# Suggest Improvements

1. We'll need to use sophisticated techniques (**HLS** color space etc.) to make it perform better under changing lighting conditions.

2. To make it work on **road turns** we'll need to use some better approach and fit a **higher degree polynomial** so that we can account for curved lines as well.