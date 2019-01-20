from main import *

if __name__ == "__main__":
    mtx, dist = calibrate_camera()
    img = cv2.imread('../test_images/straight_lines1.jpg')
    undistort = cv2.undistort(img, mtx, dist, None, mtx)
    
    binary_img = create_binary_image(img)
    M, Minv = perspective_transform(img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(binary_img, M, (img.shape[1], img.shape[0]))
    warped *= 255

    leftx, lefty, rightx, righty, out_img = find_lane_sliding_window(warped)
    left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(warped.shape, leftx, lefty, rightx, righty)
    curvature, distance = curvature_and_position(ploty, left_fit, right_fit, warped.shape[1])
    print ("Curvature of lane = ", curvature)
    print ("Car is {}m right of center".format(distance))

