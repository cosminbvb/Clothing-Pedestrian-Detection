from PIL import Image
import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', path='SavedRuns/ClothingDetection/yolov5/weights/best.pt')  # create model

model.iou = 0.40  # NMS IoU threshold (0-1)
model.conf = 0.60  # confidence threshold (0-1)

imgs = []
for i in range(1, 9):
    img = Image.open(f'Datasets/Deepfashion2/test/image/00000{i}.jpg')
    imgs.append(img)

results = model(imgs, size=640)

print(results.pandas().xyxy[0])  

# OBS: If the result bboxes are empty, retry in a yolov5 environment (installing the requirements)

