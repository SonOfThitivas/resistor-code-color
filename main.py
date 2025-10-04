import matplotlib.pyplot as plt
from skimage.filters import threshold_minimum, threshold_mean, rank, gaussian, sobel
from skimage.feature import canny
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage.morphology import rectangle
from skimage import measure
import numpy as np
import cv2

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
image_mor = cv2.erode(image_bi, kernel, iterations = 15)

# plt.imshow(image_bi, cmap="gray")
# plt.show()


# find image boarder
xStart, xEnd, yStart, yEnd = image_mor.shape[0],0,image_mor.shape[1],0

for r in range(image_mor.shape[0]):
    for c in range(image_mor.shape[1]):
        if image_mor[r][c]:
            if xStart > r:
                xStart = r
            
            if yStart > c:
                yStart = c
            
            if r > xEnd: xEnd = r
            if c > yEnd: yEnd = c

image_crop = image[xStart:xEnd, yStart: yEnd]

# plt.subplot(131)
# plt.imshow(image_bi, cmap="gray")
# plt.title("Threshold segmentation")

# plt.subplot(132)
# plt.imshow(image_mor, cmap="gray")
# plt.title("Erosion by morphology")

# plt.subplot(133)
# plt.imshow(image[xStart:xEnd, yStart: yEnd, ::-1], cmap="gray")
# plt.title("Crop from original")

# plt.show()

# averaging filter
# @adapt_rgb(each_channel)
# def avr_each(image):
#     mask_w = 5
#     return rank.mean(image, rectangle(mask_w, mask_w))

@adapt_rgb(each_channel)
def avr_each(image):
    return gaussian(image, sigma=3, preserve_range=True)

# image_blur = avr_each(image[xStart:xEnd, yStart: yEnd])
# image_blur = image_blur.astype(image.dtype)

# print(image_blur)
# plt.subplot(121)
# plt.imshow(image[xStart:xEnd, yStart: yEnd,::-1])
# plt.title("Before")

# plt.subplot(122)
# plt.imshow(image_blur[:,:,::-1])
# plt.title("After")
# plt.show()

# plt.imshow(image_crop[:,:,::-1])
# yMid = image_crop.shape[0] / 2
# x1 = image_crop.shape[1] / 5 * 1
# x2 = image_crop.shape[1] / 5 * 2
# x3 = image_crop.shape[1] / 5 * 3
# x4 = image_crop.shape[1] / 5 * 4
# plt.scatter(x1, yMid, c="red", s=10, marker=".")
# plt.scatter(x2, yMid, c="red", s=10, marker=".")
# plt.scatter(x3, yMid, c="red", s=10, marker=".")
# plt.scatter(x4, yMid, c="red", s=10, marker=".")
# plt.show()

image_gray = cv2.cvtColor(image_crop.astype(image.dtype), cv2.COLOR_BGR2GRAY)
# # image_sobel = sobel(image_gray)
image_canny = canny(image_gray, sigma=1)


# quartile seperation
yQ1 = image_gray.shape[0] // 3 * 1
yQ2 = image_gray.shape[0] // 3 * 2

imageTopCrop = image_crop[:yQ1,:]
imageBottomCrop = image_crop[yQ2:,:]

plt.subplot(121)
plt.imshow(imageTopCrop[:,:,::-1])

plt.subplot(122)
plt.imshow(imageBottomCrop[:,:,::-1])

plt.show()

# Binomal histrogram / threshold segmentation
imageTopGray = cv2.cvtColor(imageTopCrop, cv2.COLOR_BGR2GRAY)
imageTopMean = threshold_mean(imageTopGray)
imageTopBi = imageTopGray < imageTopMean
imageTopBi = imageTopBi.astype(np.uint8) * 255

imageBottonGray = cv2.cvtColor(imageBottomCrop, cv2.COLOR_BGR2GRAY)
imageBottonMean = threshold_mean(imageBottonGray)
imageBottomBi = imageBottonGray < imageBottonMean
imageBottomBi = imageBottomBi.astype(np.uint8) * 255

plt.subplot(121)
plt.imshow(imageTopBi, cmap="gray")

plt.subplot(122)
plt.imshow(imageBottomBi, cmap="gray")

plt.show()

# Erosion
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
imageTopMor = cv2.erode(imageTopBi, kernel, iterations = 5)

# Erosion
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
imageBottomMor = cv2.erode(imageBottomBi, kernel, iterations = 5)

plt.subplot(121)
plt.imshow(imageTopMor, cmap="gray")

plt.subplot(122)
plt.imshow(imageBottomMor, cmap="gray")

plt.show()

TopSeg = []

# Label connected components
labels = measure.label(imageBottomMor.astype(np.bool))

# Measure regions
regions = measure.regionprops(labels)
print(regions[0].bbox)
# Extract x,y ranges for each segment
for i, region in enumerate(regions, start=1):
    min_row, min_col, max_row, max_col = region.bbox

    # Note: rows = Y, cols = X in image coordinates
    y_min, y_max = min_row, max_row
    x_min, x_max = min_col, max_col

    print(f"Segment {i}:")
    print(f" - X range: {x_min} → {x_max}")
    print(f" - Y range: {y_min} → {y_max}")
    print()
    TopSeg.append(
        {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
        }
    )
    
for i, v in enumerate(TopSeg):
    plt.subplot(1, len(TopSeg), i+1)
    print(v)
    temp = imageTopCrop[v["y_min"]:v["y_max"],
                        v["x_min"]:v["x_max"]]
    plt.imshow(temp[:,:,::-1])

plt.show()