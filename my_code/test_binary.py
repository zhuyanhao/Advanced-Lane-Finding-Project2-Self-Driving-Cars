from main import create_binary_image
import cv2
import numpy as np

if __name__ == "__main__":
    img = cv2.imread('../test_images/straight_lines1.jpg')
    binary_img = create_binary_image(img)
    binary_img *= 255
    binary_img_to_write = np.dstack((binary_img, binary_img, binary_img))
    cv2.imwrite('straight_lines1_binary.jpg', binary_img_to_write)