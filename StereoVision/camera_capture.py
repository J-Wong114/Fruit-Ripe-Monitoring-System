import cv2
import keyboard


def capture_photo_from_cameras(camera_1, camera_2, count):
    print(f"Capturing from left and right cameras")

    photo_id = count

    file_name_right = f"stereo/right/{photo_id}.jpg"
    file_name_left = f"stereo/left/{photo_id}.jpg"

    result_left, image_left = camera_1.read()
    result_right, image_right = camera_2.read()

    if result_left:
        cv2.imwrite(file_name_left, image_left)
        print(f"Photo saves as {file_name_left}")
    else:
        print(f"No image detected from left camera")

    if result_right:
        cv2.imwrite(file_name_right, image_right)
        print(f"Photo saves as {file_name_right}")
    else:
        print(f"No image detected from right camera")


if __name__ == "__main__":
    counter = 0

    camera_id_1 = 1
    camera_id_2 = 2

    print("Loading cameras...")

    cam1 = cv2.VideoCapture(camera_id_1)
    cam2 = cv2.VideoCapture(camera_id_2)

    if not cam1.isOpened() or not cam2.isOpened():
        print("Error: Could not open both cameras.")
        cam1.release()
        cam2.release()
        exit()

    print("Cameras loaded!")

    while True:
        keyboard.read_key()
        if keyboard.is_pressed("space"):
            print("Space bar pressed")
            capture_photo_from_cameras(cam1, cam2, counter)
            counter += 1
        elif keyboard.is_pressed("esc"):
            print("Exiting...")
            cam1.release()
            cam2.release()
            cv2.destroyAllWindows()
            break
