import matplotlib.pyplot as plt
from skimage.filters import threshold_minimum, threshold_mean, rank, gaussian, sobel
from skimage.feature import canny
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage.morphology import rectangle
import numpy as np
import skimage 
import cv2
import numpy as np

# image input


# image = cv2.imread(r"img\3.9k\IMG_0525.JPG")
# image = cv2.imread(r"img\10k\IMG_0581.JPG")
image = cv2.imread(r"res.jpg")
# image = cv2.imread(r"img\56k\IMG_0550.JPG")

# image = skimage.io.imread(r"img\3.9k\IMG_0525.JPG")
# image = cv2.imread(r"img\3.9k\IMG_0541.JPG")
# image = cv2.imread(r"img\3.9k\IMG_0536.JPG")
# image = cv2.imread(r"img\3.9k\IMG_0540.jpg")
# image = cv2.resize(image_cv, (800, 600), interpolation=cv2.INTER_AREA)

#CROP
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Binomal histrogram / threshold segmentation
image_mean = threshold_mean(image_gray)
image_bi = image_gray < image_mean
image_bi = image_bi.astype(np.uint8) * 255


# Erosion
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
image_bi = cv2.erode(image_bi, kernel, iterations = 15)

# plt.imshow(image_bi, cmap="gray")
# plt.show()


# find image boarder
xStart, xEnd, yStart, yEnd = image_bi.shape[0],0,image_bi.shape[1],0

for r in range(image_bi.shape[0]):
    for c in range(image_bi.shape[1]):
        if image_bi[r][c]:
            if xStart > r:
                xStart = r
            
            if yStart > c:
                yStart = c
            
            if r > xEnd: xEnd = r
            if c > yEnd: yEnd = c




print(f"xStart: {xStart}", f"xEnd: {xEnd}", f"yStart: {xStart}", f"xEnd: {yStart}", sep="\n")
plt.imshow(image[xStart:xEnd, yStart: yEnd, ::-1])
plt.imsave( "./rescrop5.6kv2.jpg", image[xStart:xEnd, yStart: yEnd, ::-1])
# plt.show()


# averaging filter
# @adapt_rgb(each_channel)
# def avr_each(image):
#     mask_w = 5
#     return rank.mean(image, rectangle(mask_w, mask_w))

@adapt_rgb(each_channel)
def avr_each(image):
    return gaussian(image, sigma=3, preserve_range=True)

image_blur = avr_each(image[xStart:xEnd, yStart: yEnd])
image_blur = image_blur.astype(image.dtype)

# print(image_blur)
# plt.subplot(121)
# plt.imshow(image[xStart:xEnd, yStart: yEnd,::-1])
# plt.title("Before")

# plt.subplot(122)
# plt.imshow(image_blur[:,:,::-1])
# plt.title("After")
# plt.show()

image_gray = cv2.cvtColor(image[xStart:xEnd, yStart: yEnd], cv2.COLOR_BGR2GRAY)
image_sobel = sobel(image_gray)
image_canny = canny(image_gray, sigma=3)

# plt.subplot(221)
# plt.imshow(image[xStart:xEnd, yStart: yEnd,::-1])
# plt.title("Before")

# print(image_blur)
# plt.subplot(221)
# plt.imshow(image[xStart:xEnd, yStart: yEnd,::-1])
# plt.title("Before")

# plt.subplot(222)
# plt.imshow(image_blur[:,:,::-1])
# plt.title("After")

# plt.subplot(223)
# plt.imshow(image_sobel, cmap="gray")
# plt.title("Sobel")

# plt.subplot(224)
# plt.imshow(image_canny, cmap="gray")
# plt.title("Canny")
# plt.show()
