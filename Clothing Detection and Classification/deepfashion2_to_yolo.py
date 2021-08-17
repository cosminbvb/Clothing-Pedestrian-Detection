import json
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

for num in range(1, 191962):
    json_name = 'Datasets/Deepfashion2/train/annos/' + str(num).zfill(6)+'.json'
    image_name = 'Datasets/Deepfashion2/train/image/' + str(num).zfill(6)+'.jpg'
    yolo_txt_name = 'Datasets/Deepfashion2/yolo_format/train/labels/' + str(num).zfill(6)+'.txt' 
    print(num)
    imag = Image.open(image_name)
    width, height = imag.size
    with open(json_name, 'r') as f, open(yolo_txt_name, 'w') as g:
        temp = json.loads(f.read())
        for i in temp:
            if i == 'source' or i=='pair_id':
                continue
            else:
                # yolo format: class x_center y_center width height
                # bbox coordinates must be normalized (0-1)
                # class numbers are 0-indexed
                box = temp[i]['bounding_box']
                x1, y1, x2, y2 = box
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                cat = temp[i]['category_id'] - 1  # yolo indexes from 0

                g.write(f'{cat} {x_center} {y_center} {w} {h}\n')

# # visualize:
# for num in range(1, 32154):
#     image_name = 'Datasets/Deepfashion2/train/image/' + str(num).zfill(6)+'.jpg'
#     yolo_txt_name = 'Datasets/Deepfashion2/yolo_format/train/labels/' + str(num).zfill(6)+'.txt' 
#     imag = Image.open(image_name)
#     width, height = imag.size
#     # print(f'image width, height: {width}, {height}')
#     print(num)
#     fig, ax = plt.subplots(1)
#     ax.imshow(imag)
#     with open(yolo_txt_name, 'r') as f:
#         for obj in f.readlines():
#             line = [float(i) for i in obj.split()]
#             x_center = line[1] * width
#             y_center = line[2] * height
#             w = line[3] * width
#             h = line[4] * height
#             # print("center:")
#             # print(x_center, y_center)
#             # print("w, h:")
#             # print(w, h)
#             left_down = (x_center - w / 2, y_center - h / 2)
#             rect = patches.Rectangle(left_down, w, h, linewidth=1,
#                             edgecolor='r', facecolor="none")
#             ax.add_patch(rect)
#             # print('\n')
#         plt.show()
#     # print('--------------')
