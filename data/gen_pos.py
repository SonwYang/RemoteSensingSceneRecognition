import os
import cv2
import numpy as np
import sys
sys.path.append('data')
from shp2imagexy import *
import glob
import matplotlib.pyplot as plt


def bbox_to_rect(bbox, color):  # 本函数已保存在d2lzh包中方便以后使用
    # 将边界框(左上x, 左上y, 右下x, 右下y)格式转换成matplotlib格式：
    # ((左上x, 左上y), 宽, 高)
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=True, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 128), np.mod(dh, 128)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


if __name__ == '__main__':
    imglist = glob.glob('D:/2021/7/em20210628/sheepfold2/*/*.tif')
    image_size = 192
    for imgPath in imglist:
        print(imgPath)
        try:
            imgName = os.path.split(imgPath)[-1].split('.')[0]
            shpPath = imgPath.replace('tif', 'shp')
            anns = shp2imagexy(imgPath, shpPath)
            anns = [ann[:-1] for ann in anns]
            boxes = np.array(anns, dtype=np.uint16)
            img = cv2.imread(imgPath, cv2.IMREAD_LOAD_GDAL)

            # show results
            # fig = plt.imshow(img)
            # for i, box in enumerate(boxes):
                # rect = bbox_to_rect(box, 'red')
                # fig.axes.add_patch(rect)
                # fig.axes.text(rect.xy[0] + 24, rect.xy[1] + 10, "sheepfold",
                #               va='center', ha='center', fontsize=6, color='blue',
                #               bbox=dict(facecolor='m', lw=0))

            # plt.show()
            w, h = img.shape[:2]
            for i, box in enumerate(boxes):
                w0, h0 = (image_size - (box[3] - box[1])) // 2, (image_size - (box[2] - box[0])) // 2
                crop = img[np.clip(box[1]-w0, 0, w):np.clip(box[3]+w0, 0, w),
                            np.clip(box[0]-h0, 0, w):np.clip(box[2]+h0, 0, w)]
                crop, ratio, (dw, dh) = letterbox(crop, new_shape=(image_size, image_size))
                # crop = img[box[1]:box[3], box[0]:box[2]]
                # plt.subplot(121)
                # plt.imshow(crop)
                # plt.subplot(122)
                # plt.imshow(img)
                # plt.show()
                savePath = os.path.join('pos', f'{imgName}_{i}.tif')
                cv2.imwrite(savePath, crop)
        except:
            continue