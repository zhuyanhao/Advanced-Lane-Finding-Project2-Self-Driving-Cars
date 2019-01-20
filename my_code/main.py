import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import pickle
import os
from moviepy.editor import VideoFileClip

###############################################################################
## Helper Functions ###########################################################
###############################################################################
def calibrate_camera(path_to_images='../camera_cal/calibration*.jpg', pattern=(9, 6)):
    """
    Calibrate the camera using the chessboard images provided in camara_cal folder
    """
    # Prepare object points
    objp = np.zeros((pattern[1]*pattern[0],3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(path_to_images)
    img_w = 0
    img_h = 0

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        img_h, img_w, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern, None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
        
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (img_w, img_h) ,None,None)
    
    # Return the camera matrix and distortion coefs
    return mtx, dist

def sobel_mask(gray, sx_thresh):
    sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    # Normalize it so that the threshold can be more easily given
    max_value = np.max(sobel_x)
    binary_output = np.array(255*sobel_x/max_value, np.uint8)
    mask = np.zeros_like(binary_output)
    mask[(binary_output > sx_thresh[0]) & (binary_output < sx_thresh[1])] = 1
    return mask

def dir_mask(gray, thresh):
    # x_gradient and y_gradient
    # We only care about their absolute value
    sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))

    # Angle to horizontal line
    direction = np.abs(np.arctan2(sobel_x, sobel_y))

    # Mask
    mask = np.zeros_like(direction)
    mask[(direction >= thresh[0]) & (direction <= thresh[1])] = 1

    return mask

def create_binary_image(img, s_thresh=(100, 255), sx_thresh=(10, 200), dir_thresh=(np.pi/6, np.pi/2), c_thresh=50):
    """
    Create binary image by combining S channel and gradient thresholds
    I found that the video doesn't look good if I just use the filters covered in lecture.
    A better solution was found here, which combines HLS with RGB color space:
        https://github.com/subodh-malgonde/advanced-lane-finding/blob/master/Advanced_Lane_Lines.ipynb
    """
    # We use a combination of gradient and direction threshold
    # convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute the combined threshold
    sobel_x = sobel_mask(gray, sx_thresh)
    dir_gradient = dir_mask(gray, dir_thresh)
    combined = ((sobel_x == 1) & (dir_gradient == 1))

    # Color threshold in RGB color space
    # This helps to detect yellow lanes better, which is a significant issue in the video 
    G = img[:,:,1]
    R = img[:,:,2]
    r_g = (R > c_thresh) & (G > c_thresh)
    
    # color channel thresholds
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    S = hls[:,:,2]
    L = hls[:,:,1]
    
    # S channel performs well for detecting bright yellow and white lanes
    s = (S > s_thresh[0]) & (S <= s_thresh[1])
    l = (L > s_thresh[0]) & (L <= s_thresh[1])

    # combine all the thresholds
    # The pixel we want is either white or yellow
    color_combined = np.zeros_like(R)
    color_combined[(r_g & l) & (s | combined)] = 1
    
    # apply the region of interest mask
    # This helps to remove the shadow outside the lane
    mask = np.zeros_like(color_combined)
    h, w = img.shape[0], img.shape[1]
    polygon_vertice = np.array([[0,h-1], [w//2, h//2], [w-1, h-1]], dtype=np.int32)
    cv2.fillPoly(mask, [polygon_vertice], 1)
    binary = cv2.bitwise_and(color_combined, mask)
    
    return binary

def perspective_transform():
    """
    Generate the transformation (and its inverse) matrix
    """
    src = np.float32([(220,720), (1110, 720), (570, 470), (722, 470)])  # Manually get these numbers from plot
    dst = np.float32([[320, 720], [920, 720], [320, 1], [920, 1]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv

def find_lane_sliding_window(binary_warped):
    """
    Use sliding window to find the lane pixels.
    This should be used when the lane is not found in the previous frames
    """
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 10
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 100

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def fit_poly(img_shape, leftx, lefty, rightx, righty):
     ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fit, right_fit, left_fitx, right_fitx, ploty

def find_lane_from_prior(binary_warped, left_fit, right_fit, ploty):
    """
    Given the lane from previous frame, find the lane in the current frame
    """
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
    left_lane = np.array(list(zip(left_fitx, ploty)), np.int32)
    right_lane = np.array(list(zip(right_fitx, ploty)), np.int32)
    result = cv2.polylines(result, [left_lane], False, (0,255,255), 4)
    result = cv2.polylines(result, [right_lane], False, (0,255,255), 4)
    ## End visualization steps ##
    
    return leftx, lefty, rightx, righty, result

def curvature_and_position(ploty, left_fit, right_fit, img_w):
    """
    Compute the curvature of lane at the bottom and
    find the position of car relative to the center line
    """
    # Define y-value of interest
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Define some constants
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Compute curvature
    left_fit[0] = left_fit[0]*xm_per_pix/(ym_per_pix**2)
    left_fit[1] = left_fit[1]*xm_per_pix/ym_per_pix
    right_fit[0] = right_fit[0]*xm_per_pix/(ym_per_pix**2)
    right_fit[1] = right_fit[1]*xm_per_pix/ym_per_pix
    left_curverad = np.power(1+(2*left_fit[0]*y_eval+left_fit[1])**2, 3/2)/np.abs(2*left_fit[0])
    right_curverad = np.power(1+(2*right_fit[0]*y_eval+right_fit[1])**2, 3/2)/np.abs(2*right_fit[0])
    aver_curverad = 0.5*(left_curverad + right_curverad)

    # Compute the relative position
    x_left = left_fit[0]*y_eval**2+left_fit[1]*y_eval+left_fit[2]
    x_right = right_fit[0]*y_eval**2+right_fit[1]*y_eval+right_fit[2]

    car_pos = img_w // 2
    center_line = (x_left + x_right) // 2
    
    distance = (car_pos - center_line)*xm_per_pix
    
    # Return the left/right curvature and the distance to the center line (right is positive)
    return aver_curverad, distance

def draw_lane_on_img(undist, warped, left_fitx, right_fitx, ploty, Minv, curvature, distance):
    """
    Draw the left/right lane back to the original image
    """
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Draw lane boundary
    left_lane = np.int32(pts_left)
    right_lane = np.int32(pts_right)
    color_warp = cv2.polylines(color_warp, [left_lane], False, (255,0,0), 16)
    color_warp = cv2.polylines(color_warp, [right_lane], False, (0, 0, 255), 16)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.4, 0)

    # Write out the curvature and relative position
    curv_info = "Radius of Curvature ={0:.2f}(m).".format(curvature)
    if distance > 0:
        pos_info = " Car is {0:.2f}m right of center.".format(distance)
    else:
        pos_info = " Car is {0:.2f}m left of center.".format(-distance)
    info = curv_info + pos_info
    cv2.putText(result, info, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, bottomLeftOrigin=False)

    return result

def process_img(img, mtx, dist, line):
    """
    Given the original image, camera calibration properties, draw the lane lines
    on the original image and return the image
    The line class is updated on the fly.
    """
    undistort = cv2.undistort(img, mtx, dist, None, mtx)
    binary_img = create_binary_image(img)
    M, Minv = perspective_transform()
    warped = cv2.warpPerspective(binary_img, M, (img.shape[1], img.shape[0]))
    warped *= 255

    # Use line.fine_lane method to find the lane
    left_fitx, right_fitx, ploty, curvature, distance = line.find_lane(warped)

    # Draw lanes and return it
    out_img = draw_lane_on_img(img, warped, left_fitx, right_fitx, ploty, Minv, curvature, distance)
    return out_img
    

###############################################################################
## Helper Class ###############################################################
###############################################################################
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False

        # poly fit from previous frame
        self.last_left_fit = None
        self.last_right_fit = None

        # Tolerance and number of frame to average
        self.tol_dist = 4

    def find_lane(self, warped_img):
        """
        Find the lane in image
        The lane is rejected if it is rapidly changed from previous frame;
        otherwise it is averaged in the latest n frames
        """
        if not self.detected:
            # Use sliding window if lanes are not detected on the previous frame
            leftx, lefty, rightx, righty, out_img = leftx, lefty, rightx, righty, out_img = find_lane_sliding_window(warped_img)
            left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(warped_img.shape, leftx, lefty, rightx, righty)
            curvature, distance = curvature_and_position(ploty, left_fit, right_fit, warped_img.shape[1])
            self.last_left_fit = left_fit
            self.last_right_fit = right_fit
            self.last_ploty = ploty
            self.detected = True
            return left_fitx, right_fitx, ploty, curvature, distance
        
        else:
            try:
                leftx, lefty, rightx, righty, out_img = find_lane_from_prior(warped_img, self.last_left_fit, self.last_right_fit, self.ploty)
                left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(warped_img.shape, leftx, lefty, rightx, righty)
                curvature, distance = curvature_and_position(ploty, left_fit, right_fit, warped_img.shape[1])
                
                # If the distance doesn't make sense, use sliding window to search again
                if abs(distance) > self.tol_dist:
                    self.detected = False
                    return self.find_lane(warped_img)
                else:
                    self.last_left_fit = left_fit
                    self.last_right_fit = right_fit
                    self.last_ploty = ploty
                    self.detected = True
                    return left_fitx, right_fitx, ploty, curvature, distance
            except:
                # Exception raised by fitpoly when left/right is empty
                self.detected = False
                return self.find_lane(warped_img)

###############################################################################
## Functions called in main ###################################################
###############################################################################
def process_all_images(path=r'../test_images'):
    # Iterate through all files
    for filename in os.listdir(path):
        mtx, dist = calibrate_camera()
        path_to_image = os.path.join(path, filename)
        img = cv2.imread(path_to_image)
        new_img = process_img(img, mtx, dist, Line())
        new_filename = os.path.splitext(filename)[0] + "_output.jpg"
        new_filename = os.path.join('../output_images', new_filename)
        cv2.imwrite(new_filename, new_img)

def process_videos(path_to_video, path_to_output):
    # Create a wrapper on process_img function
    mtx, dist = calibrate_camera()
    line = Line()
    def process_video_frame(img):
        return process_img(img, mtx, dist, line)
    
    clip = VideoFileClip(path_to_video)
    new_clip = clip.fl_image(process_video_frame)
    new_clip.write_videofile(path_to_output, audio=False)

if __name__ == "__main__":
    process_all_images()
    process_videos('../project_video.mp4', '../output_videos/project_video_output.mp4')
    process_videos('../challenge_video.mp4', '../output_videos/challenge_video.mp4')
    # process_videos('../harder_challenge_video.mp4', '../output_videos/harder_challenge_video.mp4')
