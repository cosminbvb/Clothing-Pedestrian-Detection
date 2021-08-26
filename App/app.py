import os
import io
import base64
import copy

# app imports
from flask import Flask, redirect, url_for, render_template, request, flash, jsonify
from werkzeug.utils import secure_filename

# ml imports
import numpy as np
import torch
from PIL import Image
import torchvision
import torchvision.transforms as T
import cv2

# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.utils.visualizer import ColorMode
# from detectron2.data import DatasetCatalog, MetadataCatalog

UPLOAD_FOLDER = 'static/storage_testing'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


app = Flask(__name__)
app.secret_key = "dev"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['image']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            file = file.read()  # bytes
            np_img = np.fromstring(file, np.uint8)
            img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)  # BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert

            people = detect_people(img)  # people is now an array of cv2 images, each one containing 1 detected person
            people_copy = copy.deepcopy(people)

            # clothing = detect_clothing_detectron2(people_copy)  # detectron2 instance segmentation
            if len(people) > 0:
                clothing = detect_clothing_yolo(people_copy)  # yolov5 object detection
            else:
                clothing = []
            # clothing is now an array of cv2 images, each one showing the clothing items for each person

            result_images = people + clothing

            return_dict = {}
            return_dict['status'] = len(result_images)  # nr of images to send 
            for i, image in enumerate(result_images):
                img = Image.fromarray(image.astype('uint8')).convert('RGB')
                # img.save(os.path.join(app.config['UPLOAD_FOLDER'], f'person_pil_{i}.jpg'))
                rawBytes = io.BytesIO()
                img.save(rawBytes, "JPEG")
                rawBytes.seek(0)
                img_base64 = base64.b64encode(rawBytes.read())
                return_dict[f'image_{i}'] = str(img_base64)
            return jsonify(return_dict)
        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)
    else:
        return render_template('index.html')
        

def detect_people(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_copy = image.copy()  # used later while cropping

    preprocess = T.Compose([
        # T.Resize(some value),
        T.ToTensor()
    ])

    image = preprocess(image)
    
    model = torch.load('../Clothing Detection and Classification/SavedRuns/PersonDetection/pytorch_faster_rcnn/100epochs.pt')
    model.to(device)
    model.eval()
    with torch.no_grad():
        prediction = model([image.to(device)])
    boxes = prediction[0]["boxes"]
    scores = prediction[0]["scores"]
    keep = torchvision.ops.nms(boxes, scores, 0.5) 
    boxes = boxes.tolist()
    scores = scores.tolist()
    
    threshold = 0.6
    final_boxes = []
    final_scores = []

    people = []
    for i in keep:
        if scores[i] < threshold:
            continue
        else:
            final_boxes.append(boxes[i])
            final_scores.append(scores[i])
            x1, y1, x2, y2 = map(int, boxes[i])
            cropped = image_copy[y1:y2, x1:x2]
            people.append(cropped)
    
    print(final_boxes)
    print(final_scores)

    return people


def detect_clothing_yolo(people):
    imgs = []
    for person in people:
        # decided to add padding because the labels drawn by yolo usually don't fit into the image
        padded = cv2.copyMakeBorder(person, top=0, bottom=0, left=0, right=250,
                                    borderType=cv2.BORDER_CONSTANT, value=[16, 21, 24])
        imgs.append(Image.fromarray(padded.astype('uint8')))
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='../Clothing Detection and Classification/SavedRuns/ClothingDetection/yolov5/weights/best.pt')  # create model
    model.conf = 0.6  # confidence threshold (0-1)
    results = model(imgs, size=640)
    results.render()
    return results.imgs
    

def detect_clothing_detectron2(people):
    output_images = []
    cfg = get_cfg()
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = '../Clothing Detection and Classification/SavedRuns/ClothingDetection/detectron2_maskrcnn/model_final.pth'  
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.65  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    my_dataset_val_metadata = MetadataCatalog.get("DeepFashion2_valid")
    for i, person in enumerate(people):
        outputs = predictor(person)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(person[:, :, ::-1],
                    metadata=my_dataset_val_metadata, 
                    scale=1, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        output_image = cv2.cvtColor(out.get_image(), cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], f'person_{i}.jpg'), output_image)
        output_images.append(output_image)
    return output_images


if __name__ == "__main__":
    app.run(debug=True)

