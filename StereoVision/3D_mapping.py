import cv2
import numpy as np
import open3d as o3d

# Intrinsics
K_left = np.array([[1077.5, 0.0, 950.3],
                   [0.0, 1077.9, 511.7],
                   [0.0, 0.0, 1.0]])

dist_left = np.array([-0.0005, -0.0411, 0.0, 0.0, 0.0])

K_right = np.array([[1045.4, 0.0, 957.1],
                    [0.0, 1041.4, 541.3],
                    [0.0, 0.0, 1.0]])

dist_right = np.array([-0.0382, -0.0144, 0.0, 0.0, 0.0])

# Extrinsics from stereo calibration
R = np.array([[9.99586896e-01, -8.23331887e-04, -2.87290566e-02],
              [5.75493128e-04,  9.99962561e-01, -8.63396514e-03],
              [2.87350896e-02,  8.61386505e-03,  9.99549947e-01]])

T = np.array([[-0.08224825],
              [0.00634638],
              [-0.03288949]])

##################
#  Camera setup  #
##################
print("Loading Cameras...")
camera_id_1 = 1  # Camera A (Left)
camera_id_2 = 2  # Camera B (Right)

capL = cv2.VideoCapture(camera_id_1)
capR = cv2.VideoCapture(camera_id_2)

if not capL.isOpened() or not capR.isOpened():
    raise RuntimeError("Could not open both cameras")

print("Cameras Sucessfully Loaded!")

#########################
#  Rectification setup  #
#########################
ret, frameL = capL.read()
image_size = (frameL.shape[1], frameL.shape[0])

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    K_left, dist_left,
    K_right, dist_right,
    image_size,
    R, T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0
)

mapL1, mapL2 = cv2.initUndistortRectifyMap(
    K_left, dist_left, R1, P1, image_size, cv2.CV_16SC2)

mapR1, mapR2 = cv2.initUndistortRectifyMap(
    K_right, dist_right, R2, P2, image_size, cv2.CV_16SC2)

####################
#  Stereo matcher  #
####################
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*10,
    blockSize=5,
    P1=8 * 3 * 5**2,
    P2=32 * 3 * 5**2,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    disp12MaxDiff=1,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

################################
#  Open3D visualization setup  #
################################
print("3D Visualization Setup...")
pcd = o3d.geometry.PointCloud()
vis = o3d.visualization.Visualizer()
vis.create_window("3D Stereo Map")

added = False
print("3D Visualization Setup has Completed!")

##############
#  Main Loop #
##############
print("Beginning Main Loop")
while True:
    retL, imgL = capL.read()
    retR, imgR = capR.read()
    if not retL or not retR:
        break

    rectL = cv2.remap(imgL, mapL1, mapL2, cv2.INTER_LINEAR)
    rectR = cv2.remap(imgR, mapR1, mapR2, cv2.INTER_LINEAR)

    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    # Compute Disparity
    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

    # Apply a "Closing" filter to fill small holes in the disparity map
    kernel = np.ones((5, 5), np.uint8)
    disparity = cv2.morphologyEx(disparity, cv2.MORPH_CLOSE, kernel)

    # Reproject disparity to 3D
    points_3D = cv2.reprojectImageTo3D(disparity, Q)

    # Get Colors
    colors = cv2.cvtColor(rectL, cv2.COLOR_BGR2RGB)

    # ---- DEPTH FILTERING ----
    Z = points_3D[:, :, 2]

    # Mask: Disparity must be valid, and depth must be within range
    mask = (disparity > disparity.min()) & (Z > 0.2) & (Z < 2.0)

    pts = points_3D[mask]
    cols = colors[mask] / 255.0

    # Case where no points are found
    if len(pts) > 0:
        # Update the EXISTING pcd object. Do not create a new one.
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)

        if not added:
            vis.add_geometry(pcd)
            added = True
        else:
            vis.update_geometry(pcd)

    vis.poll_events()
    vis.update_renderer()

    # ---- VISUAL DEBUGGING ---- #
    # Normalize disparity for visualization (0-255)
    disp_vis = (disparity - disparity.min()) / \
        (disparity.max() - disparity.min())
    cv2.imshow("Disparity Map (Debug)", disp_vis)
    cv2.imshow("Left Rectified", rectL)

    if cv2.waitKey(1) & 0xFF == 27:
        break

###########
# Cleanup #
###########
capL.release()
capR.release()
cv2.destroyAllWindows()
vis.destroy_window()
