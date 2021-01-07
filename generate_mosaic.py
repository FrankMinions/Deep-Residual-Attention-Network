import os
import cv2 as cv

alpha = 5
image_list = os.listdir('./test_raw/')
# image_list = os.listdir('./train_raw/')

for k in range(len(image_list)):
    img = cv.imread(os.path.join('./test_raw/', image_list[k]))
    # img = cv.imread(os.path.join('./train_raw/', image_list[k]))

    height, width, deep = img.shape

    for m in range(0, height, alpha):
        for n in range(0, width, alpha):
            b, g, r = img[m, n]
            img[m:m+alpha, n:n+alpha, :] = (b, g, r)

    cv.imwrite(os.path.join('./test_mosaic/', image_list[k][:image_list[k].rfind('.')]+'.png'), img)
    # cv.imwrite(os.path.join('./train_mosaic/', image_list[k][:image_list[k].rfind('.')]+'.png'), img)