import cv2
import numpy as np

class StereoVision:
    def __init__(self, cam_id_left=2, cam_id_right=0):
        # --- NATIVE 640x480 CALIBRATION DATA ---
        self.K_left = np.array([[305.5628900971274, 0.0, 322.6041545860787], 
                                [0.0, 306.64364937926257, 256.1733414900114], 
                                [0.0, 0.0, 1.0]])
        self.dist_left = np.array([-0.12481987274029067, 0.03980170877665275, 0.006890465284692507, -0.012330388102439369, -0.009113841478988353])
        
        self.K_right = np.array([[299.62120071243294, 0.0, 342.320419416251], 
                                 [0.0, 297.18050202353567, 222.2019245812153], 
                                 [0.0, 0.0, 1.0]])
        self.dist_right = np.array([-0.025967798516161808, -0.054511048775194006, 0.014730536995968484, 0.0014886880174935285, 0.026175180162268847])
        
        self.R = np.array([[0.9944041666178655, 0.018646433241242752, -0.10398396001503486], 
                           [-0.012955498682458544, 0.9983946734707316, 0.05513829013634702], 
                           [0.104845164251468, -0.05348258139479661, 0.9930493970696682]])
        self.T = np.array([[-0.10], [0.03943064258230158], [0.0020826940897071006]])
        # self.T = np.array([[-0.0914561711041949], [0.03943064258230158], [0.0020826940897071006]])


        # --- Initialize Cameras ---
        print(f"[INFO] Initializing Stereo Pair: Left={cam_id_left}, Right={cam_id_right}")
        self.capL = cv2.VideoCapture(cam_id_left, cv2.CAP_V4L2)
        self.capR = cv2.VideoCapture(cam_id_right, cv2.CAP_V4L2)
        
        for cam in [self.capL, self.capR]:
            cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cam.set(cv2.CAP_PROP_FPS, 30)

        if not self.capL.isOpened() or not self.capR.isOpened():
             print("[ERROR] Could not open one or both cameras.")
             return

        self.image_size = (640, 480)
        self.init_rectification_maps()
        
        # --- Stereo SGBM Matcher ---
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16*8,  # 128 max disparity, great for 640x480 resolution
            blockSize=11,         
            P1=8 * 3 * 11**2,
            P2=32 * 3 * 11**2,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            disp12MaxDiff=1,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

    def init_rectification_maps(self):
        R1, R2, P1, P2, self.Q, _, _ = cv2.stereoRectify(
            self.K_left, self.dist_left,
            self.K_right, self.dist_right,
            self.image_size, self.R, self.T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )
        self.mapL1, self.mapL2 = cv2.initUndistortRectifyMap(
            self.K_left, self.dist_left, R1, P1, self.image_size, cv2.CV_16SC2)
        self.mapR1, self.mapR2 = cv2.initUndistortRectifyMap(
            self.K_right, self.dist_right, R2, P2, self.image_size, cv2.CV_16SC2)

    def get_frames(self):
        retL, imgL = self.capL.read()
        retR, imgR = self.capR.read()
        if not retL or not retR:
            return None, None, None, None
        
        rectL = cv2.remap(imgL, self.mapL1, self.mapL2, cv2.INTER_LINEAR)
        rectR = cv2.remap(imgR, self.mapR1, self.mapR2, cv2.INTER_LINEAR)
        return imgL, imgR, rectL, rectR

    def get_object_depth(self, rectL, rectR, bbox):
        x, y, w, h = bbox
        
        # Bounds check
        x = max(0, x); y = max(0, y)
        w = min(self.image_size[0] - x, w)
        h = min(self.image_size[1] - y, h)

        grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

        # Compute raw disparity
        disparity = self.stereo.compute(grayL, grayR).astype(np.float32) / 16.0
        
        # --- 3D REPROJECTION ---
        points_3D = cv2.reprojectImageTo3D(disparity, self.Q)
        roi_3d = points_3D[y:y+h, x:x+w]

        # Filter for valid depth range (10cm to 3m)
        Z_values = roi_3d[:, :, 2]
        mask = (Z_values > 0.1) & (Z_values < 3.0) & np.isfinite(Z_values)
        
        if np.count_nonzero(mask) < 10:
            return None 

        valid_points = roi_3d[mask]
        
        # Use median to filter out outliers/noise
        median_x = np.median(valid_points[:, 0])
        median_y = np.median(valid_points[:, 1])
        median_z = np.median(valid_points[:, 2])

        return (median_x, median_y, median_z)

    def release(self):
        if self.capL: self.capL.release()
        if self.capR: self.capR.release()
