import matplotlib.pyplot as plt
import matplotlib.image as mpimg

if __name__ == "__main__":
    img=mpimg.imread('../test_images/straight_lines1.jpg')
    imgplot = plt.imshow(img)
    plt.show()