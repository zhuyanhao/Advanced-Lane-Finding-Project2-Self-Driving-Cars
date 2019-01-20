from main import calibrate_camera
import cv2

if __name__ == "__main__":
    mtx, dist = calibrate_camera()
    
    # Read in a image and undistort it
    img = cv2.imread('../camera_cal/calibration4.jpg')
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('calibration4_undistorted.jpg',dst)