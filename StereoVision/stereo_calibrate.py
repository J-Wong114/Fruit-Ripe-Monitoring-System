import cv2
import numpy as np
import glob

#####################
# Camera Parameters #
#####################

# Left camera
K_left = np.array([[1077.5, 0.0, 950.3],
                   [0.0, 1077.9, 511.7],
                   [0.0, 0.0, 1.0]], dtype=np.float64)

dist_left = np.array([-0.0005, -0.0411, 0.0, 0.0, 0.0], dtype=np.float64)

# Right camera
K_right = np.array([[1045.4, 0.0, 957.1],
                    [0.0, 1041.4, 541.3],
                    [0.0, 0.0, 1.0]], dtype=np.float64)

dist_right = np.array([-0.0382, -0.0144, 0.0, 0.0, 0.0], dtype=np.float64)


######################
# Stereo Claibration #
######################

# Chessboard info
chessboard_size = (9, 6)
square_size = 0.022  # meters


objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints_left = []
imgpoints_right = []

# Load corresponding checkerboard image pairs
left_images = sorted(glob.glob("stereo/left/*.jpg"))
right_images = sorted(glob.glob("stereo/right/*.jpg"))

for left_img, right_img in zip(left_images, right_images):
    print(f"Checking images {str(left_img)} and {str(right_img)}")
    imgL = cv2.imread(left_img)
    imgR = cv2.imread(right_img)

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCorners(grayL, chessboard_size)
    retR, cornersR = cv2.findChessboardCorners(grayR, chessboard_size)

    if retL and retR:
        objpoints.append(objp)
        imgpoints_left.append(cornersL)
        imgpoints_right.append(cornersR)

image_size = grayL.shape[::-1]

ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpoints_left,
    imgpoints_right,
    K_left, dist_left,
    K_right, dist_right,
    image_size,
    flags=cv2.CALIB_FIX_INTRINSIC
)

print("Rotation R:\n", R)
print("Translation T (meters):\n", T)
