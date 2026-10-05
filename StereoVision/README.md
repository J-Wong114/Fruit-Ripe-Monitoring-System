# Stereo Vision
This directory contains the stereo vision pipeline that was developed as part of the Fruit Ripe Monitoring System capstone project.

The stereo vision work was developed and evaluated in several stages. The initial work focused on calibrating a pair of USB cameras and determining whether the stereo setup could produce useful 3D information. Stereo vision was later incorporated into an early robotic prototype to estimate the 3D position of detected apples.

**Note:** Stereo vision was evaluated during the development of the Fruit Ripe Monitoring System project but was ultimately not utilized in the final implementation. Reflective apple surfaces and visually similar regions made reliable stereo matching difficult. Instead, the final implementation used a monocular depth estimation approach.

# Files

| File | Purpose |
| --- | --- |
| **camera_capture.py**    | Captures checkerboard images from two cameras for camera calibration |
| **stereo_calibrate.py**  | Uses the captured checkerboard images and camera intrinsic parameters to calculate the relative rotation and translation between two cameras |
| **3D_mapping.py**        | Tests the calibrated stereo setup by generating disparity maps and reconstructing a 3D point cloud using Open3D |
| **StereoVision_v3.py**   | Stereo vision implementation used during testing of the robotic prototype. It estimates the 3D position of an object from its bounding box |

# Development Process
**1. Camera Calibration**

**camera_capture.py** was used to capture images of a checkerboard from both cameras. These image pairs were used to establish the calibration parameters required for stereo vision.

Individual camera calibration was performed using MATLAB. The resulting camera intrinsic parameters were then used by **stereo_calibrate.py**, which used corresponding checkerboard images from the two cameras to determine their relative stereo configuration.

The stereo calibration produced the rotation matrix (R) and translation vector (T) describing the relationship between the two cameras.

**2. 3D Mapping Test**

After calibration, **3D_mapping.py** was used to test the stereo camera setup independently.

This script performs:
- Stereo image rectification
- StereoSGBM disparity estimation
- 3D reprojection
- Depth filtering
- Point cloud generation and visualization using Open3D

This stage was used to determine whether the two-camera setup could successfully reproduce a 3D representation of the scene.

**3. Robotic Prototype**

After testing the stereo setup, **StereoVision_v3.py** was used during testing of the robotic prototype. This implementation captures frames from the two cameras, rectifies them using stereo calibration parameters, and calculates a disparity map using StereoSGBM. Given an object's bounding box, it extracts valid 3D points within that region and uses the median XYZ coordinates to estimate the object's 3D position.

**StereoVision_v3.py** was called by a separate **main.py** program that controlled the robot through a state-machine based pipeline. That **main.py** file was not written by me and is therefore not included in this repository.

# Camera Configuration

During development, the camera configuration was changed from a higher-resolution setup to 640 × 480 for the later robotic prototype.

**StereoVision_v3.py** explicitly configures both cameras to 640 × 480 and contains calibration parameters corresponding to that configuration.

The earlier calibration and 3D mapping scripts contain a separate set of camera parameters from an earlier development stage. These parameters have been preserved in their original scripts rather than combined with the later prototype configuration.

# Stereo Vision Approach

The stereo system used two cameras positioned at different viewpoints. After calibration and rectification, corresponding features between the left and right images could be used to calculate disparity. Disparity was then converted into 3D coordinates using the stereo camera geometry.

For the robotic prototype, the estimated 3D coordinates within an object's bounding box were summarized using the median of the valid XYZ coordinates to reduce the effect of noisy measurements and outliers.

# Limits and Final Design
Stereo vision was ultimately not chosen for the final harvesting system.

In testing, reflective apple surfaces and visually similar regions made it difficult for stereo matching to consistently identify corresponding features between the two cameras. This resulted in unreliable disparity and depth measurements for some apples.

The project transitioned to a monocular depth estimation approach for the final system.
