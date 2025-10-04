import matplotlib.pyplot as plt
import skimage as ski
import numpy as np

# from pillow_heif import read_heif
# from PIL import Image

import cv2
import numpy as np

# image = cv2.imread(r"img\3.9k\IMG_0525.jpg")
# image = cv2.imread(r"img\3.9k\IMG_0541.JPG")
image = cv2.imread(r"img\3.9k\IMG_0540.jpg")
# image = cv2.resize(image_cv, (800, 600), interpolation=cv2.INTER_AREA)

def func(i=1):
    #CROP
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # size = 100
    # image_crop = image_gray


    # # image filter
    # mask_w = 5
    # # image_denoise = filters.rank.maximum(image_noise, disk(mask_w))
    # image_denoise = ski.filters.rank.minimum(image_crop, ski.morphology.rectangle(mask_w, mask_w))


    # threshold segmentation
    # thresh_min = ski.filters.threshold_mean(image_crop)
    # binary_min = image_crop > thresh_min


    # image_bi = image_denoise < 220
    # image_bi = image_bi.astype(np.uint8) * 255
    
    
    image_gray = np.array(image_gray)
    image_mean =  np.mean(image_gray)
    image_bi = image_gray < image_mean
    image_bi = image_bi.astype(np.uint8) * 255
    

    # Erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    image_bi = cv2.erode(image_bi, kernel, iterations = i)


    # plt.imshow(image_bi, cmap="gray")
    # plt.show()


    # find image boarder
    # xStart, xEnd, yStart, yEnd = image_bi.shape[0],0,image_bi.shape[1],0

    # for r in range(image_bi.shape[0]):
    #     for c in range(image_bi.shape[1]):
    #         if image_bi[r][c]:
    #             if xStart > r:
    #                 xStart = r
                
    #             if yStart > c:
    #                 yStart = c
                
    #             if r > xEnd: xEnd = r
    #             if c > yEnd: yEnd = c
    
    return image_bi
                
# print(f"xStart: {xStart}", f"xEnd: {xEnd}", f"yStart: {xStart}", f"xEnd: {yStart}", sep="\n")
# plt.imshow(image[xStart:xEnd, yStart: yEnd, ::-1])
# plt.show()

if __name__ == "__main__":
    for i in range(20):
        img = func(i+1)
        plt.imsave(f"./morph-result/interation{i+1}.png", img, cmap="gray")
        plt.imshow(img, cmap="gray")
        plt.title(f"it = {i+1}"), plt.xticks([]), plt.yticks([])
        print("iteraton:", i+1)
        