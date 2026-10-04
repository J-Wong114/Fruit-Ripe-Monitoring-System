# Fruit Ripe Monitoring System
Mapping the real-world coordinates of apples using two USB cameras.

## Project Overview
As part of our final year capstone project, my team and I were developed an autonomous fruit detection system developed for harvesting ripe apples. To achieve this we integrated a 6 D.O.F robotic arm, Arduino UNO, Raspberry Pi 4, and two USB cameras to detect, map, and pick apples. 

This repository focuses on my contributions to the project's computer vision implementation, specifically YOLOv5n object detection and 3D stereo vision.

## YOLOv5n Object Detection
The fruit detection system implements the You Only Look Once (YOLO) architecture to identify apples and classify them as ripe, unripe, and damaged. 

I implemented the YOLOv5n (nano) model, which was chosen for its speed and size since the model was run off of a Raspberry Pi 4. 

The model was trained on dataset of 10,079 apple images obtained on Roboflow.  Data augmentation techniques, including image mirroring, brightness adjustments, and artificial overexposure, were applied to increase the diversity of the training data. Following augmentation, the dataset consisted of over 21,000 images.

<img width="420" height="420" alt="train_batch0" src="https://github.com/user-attachments/assets/146faf27-94e3-4b2a-aebe-5428291e2d14" />

Following training, the model was exported to the ONNX (Open Neural Network Exchange) format for integration into a Python-based inference pipeline. This allowed the model to be deployed using ONNX Runtime for CPU-based inference on the Raspberry Pi 4.

## 3D Stereo Vision
To determine the real-world positions of detected apples, I implemented a 3D stereo vision system using two USB cameras.

Stereo vision systems estimate depth by mimicking the binocular vision that humans have by using two cameras situated on the same plane to reconstruct the environment that they are looking at. They are able to calculate depth within an image by comparing the horizontal pixel differences of two points of interest.

<img width="2203" height="1089" alt="image" src="https://github.com/user-attachments/assets/c21dbfc4-c68a-433b-af95-e94d9ac5bc53" />

## Camera Calibration and Rectification
The intrinsic parameters and lens distortion coefficients for each USB camera were obtained individually using MATLAB's Single Camera Calibrator App.

To establish the geometric relationship between the two cameras, corresponding checkerboard image pairs were captured at different positions and orientations. OpenCV's stereoCalibrate() function was then used with the fixed intrinsic parameters to estimate the relative rotation and translation between the cameras.

These calibration parameters were subsequently used for stereo rectification and 3D reconstruction.

## Disparity Estimation and 3D Reconstruction
To calculate depth, I implemented the Semi-Global Block Matching (SGBM) algorithm using OpenCV's StereoSGBM implementation. This algorithm estimates disparity by matching corresponding regions between the rectified left and right camera images.

The resulting disparity map was converted into 3D coordinates using OpenCV's reprojectImageTo3D() function and the stereo reprojection matrix. This generated a 3D representation of the observed scene.

For object position estimation, the reconstructed 3D points within a detected apple's bounding box were filtered to remove invalid depth measurements and outliers. The median X, Y, and Z coordinates were then calculated to estimate the apple's real-world position.

## Stereo Vision Prototype
An experimental stereo vision pipeline developed for estimating the 3D position of detected apples. The implementation uses calibrated stereo cameras, image rectification, StereoSGBM disparity estimation, and 3D reprojection to extract median XYZ coordinates within an object's bounding box.

## Limitations and Final Implementation
Although the stereo vision implementation successfully demonstrated 3D reconstruction, it had difficulties reliably matching corresponding points on apples due to their smooth, reflective, and similar surfaces.

As a result, the final system did not use stereo vision. Instead we chose to use a monocular depth estimation approach to calculate the approximate 3D coordinates from the detected apple's bounding box and assumed real-world size.
