from main import perspective_transform
import cv2

if __name__ == "__main__":
    img = cv2.imread('straight_lines1_binary.jpg')
    M, _ = perspective_transform()
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    cv2.imwrite('straight_lines1_binary_warped.jpg', warped)

