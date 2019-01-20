from main import *

if __name__ == "__main__":
    mtx, dist = calibrate_camera()
    img = cv2.imread('../test_images/straight_lines1.jpg')
    line = Line()

    # Use sliding window
    img1 = process_img(img, mtx, dist, line)
    cv2.imwrite('process_img_sliding_window.jpg', img1)
    # Find from prior
    img2 = process_img(img, mtx, dist, line)
    cv2.imwrite('process_img_from_prior.jpg', img2)
