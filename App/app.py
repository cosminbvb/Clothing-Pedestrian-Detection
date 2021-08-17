import os
import io
import base64

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

            people = detect_people(img)
            detect_clothing(people)  # IN PROGRESS

            return_dict = {}
            return_dict['status'] = len(people)  # nr of people detected 
            for i, person in enumerate(people):
                img = Image.fromarray(person.astype('uint8'))
                rawBytes = io.BytesIO()
                img.save(rawBytes, "JPEG")
                rawBytes.seek(0)
                img_base64 = base64.b64encode(rawBytes.read())
                return_dict[f'person_{i}'] = str(img_base64)
            return jsonify(return_dict)
        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)
    else:
        return render_template('index.html')
        

def detect_people(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_copy = image.copy()  # used later while for crop

    preprocess = T.Compose([
        # T.Resize(some value),
        T.ToTensor()
    ])

    image = preprocess(image)
    
    model = torch.load('../Clothing Detection and Classification/SavedRuns/PersonDetection/pytorch_faster_rcnn/50_0.pt')
    model.to(device)
    model.eval()
    with torch.no_grad():
        prediction = model([image.to(device)])
    boxes = prediction[0]["boxes"]
    scores = prediction[0]["scores"]
    keep = torchvision.ops.nms(boxes, scores, 0.2) 
    boxes = boxes.tolist()
    scores = scores.tolist()
    
    threshold = 0.5
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
            people.append(image_copy[y1:y2, x1:x2].copy())

    # # for testing    
    # for i, img in enumerate(people):
    #     cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], f'person_{i}.jpg'), 
    #                 cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # at this point we detected the people in the image
    
    return people


def detect_clothing(people):
    imgs = []
    for person in people:
        imgs.append(Image.fromarray(person.astype('uint8')))
        Image.fromarray(person.astype('uint8'))
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='../Clothing Detection and Classification/SavedRuns/ClothingDetection/yolov5/weights/best.pt')  # create model
    # model.conf = 0.65  # confidence threshold (0-1)
    results = model(imgs, size=640)
    print(results.pandas().xyxy[0])


if __name__ == "__main__":
    app.run(debug=True)


# TODO
# fix the flash messages
# dupa ce se da click pe detect, imaginea selectata sa nu dispara
