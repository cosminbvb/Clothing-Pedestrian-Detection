# Clothing Recognition

This repository combines the following tasks into a fully working Clothing Recognition app:
* [Deployment with Flask](#deployment)
* Clothing detection
  * [Dataset](#ds1)
  * [YOLOv5](#ap1)
  * [Mask RCNN with Detectron2](#ap2)
  * [Faster RCNN with PyTorch](#ap3)
* [Person detection](#person-detection)
 
<a name="deployment"/>

## Deployment with Flask
![](https://github.com/cosminbvb/Clothing-Recognition/blob/main/demo.gif)


<a name="clothing-recognition"/>

## Clothing Recognition

***Disclaimer*** - This is not a fair comparison between the 3 approaches. Since the dataset is really big, a lot of time and resources 
would be required to find the best hyperparameters and reach convergence on each one of them.

<a name="ds1"/>

### Dataset
The dataset used is Deepfashion2. Find instructions on downloading it and lots of details [here](https://github.com/switchablenorms/DeepFashion2). <br>
Also, I provided the code necessary for converting the dataset into **coco** and **yolo** formats. 

<a name="ap1"/>

### YOLOv5
Check out the [official repository](https://github.com/ultralytics/yolov5). If you only want to run inference, there is no need to clone their repo since we
are using PyTorch Hub, but is **highly recommended** to run it in a virtual environment with the installed requirements, which you can find in their repo. 

Let's jump into the details:

First of all, in order to train on custom data, you will need to create a .yaml file describing the dataset.

```yaml
path: ../Clothing Classification/Datasets/Deepfashion2/yolo_format
train: train/images
val: validation/images
test: # todo

# Classes
nc: 13
names: ['short sleeve top', 'long sleeve top', 'short sleeve outwear', 'long sleeve outwear',
'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short sleeve dress', 'long sleeve dress',
'vest dress', 'sling dress']
```
Secondly, the labels need to be converted into YOLO format. You can do so with [this piece of code](https://github.com/cosminbvb/Clothing-Recognition/blob/main/Clothing%20Detection%20and%20Classification/deepfashion2_to_yolo.py).

Make sure to organise the directories in such way that the train and validation directories will each contain 2 directories called images and labels.

Then, you will have to choose a model to start training from. In my case, I went with the YOLOv5x configuration. 
As for the hyperparameters, they can be found [here](https://github.com/cosminbvb/Clothing-Recognition/blob/main/Clothing%20Detection%20and%20Classification/SavedRuns/ClothingDetection/yolov5/hyp.yaml).

Finally, I started training for 100 epochs with a batch size of 64 using the following command:
```
--img 640 --batch 64 --epochs 100 --data deepfashion2.yaml --weights yolov5x.pt
```


Now, let's take a look at how the model performed:

Results summary:
![Results](https://github.com/cosminbvb/Clothing-Recognition/blob/main/Clothing%20Detection%20and%20Classification/SavedRuns/ClothingDetection/yolov5/results.png?raw=true)

Confusion matrix:
![Conf matrix](https://github.com/cosminbvb/Clothing-Recognition/blob/main/Clothing%20Detection%20and%20Classification/SavedRuns/ClothingDetection/yolov5/confusion_matrix.png?raw=true)

Precision-Recall curve:
![PR curve](https://github.com/cosminbvb/Clothing-Recognition/blob/main/Clothing%20Detection%20and%20Classification/SavedRuns/ClothingDetection/yolov5/PR_curve.png?raw=true)

F1 curve:
![F1 curve](https://github.com/cosminbvb/Clothing-Recognition/blob/main/Clothing%20Detection%20and%20Classification/SavedRuns/ClothingDetection/yolov5/F1_curve.png?raw=true)

The saved model can be found [here](https://github.com/cosminbvb/Clothing-Recognition/tree/main/Clothing%20Detection%20and%20Classification/SavedRuns/ClothingDetection/yolov5/weights).

<a name="ap2"/>

### Instance Segmentation (Mask RCNN) with Detectron2

First of all, in order to train on a custom dataset, we'll need to register the dataset. The easiest way to do so, in my opinion, is to [convert the dataset into **coco** format](https://github.com/cosminbvb/Clothing-Recognition/blob/main/Clothing%20Detection%20and%20Classification/deepfashion2_to_coco.py) and then just run the ```register_coco_instances``` function they provide, passing the json and images directory paths as arguments.

Next, we need to define the configuration. Here, I chose the mask_rcnn_R_101_FPN_3x model configuration, 16 images per batch, 0.001 learning rate and 30k max interations.

Here is how the model performed:

![](https://github.com/cosminbvb/Clothing-Person-Detection/blob/main/Clothing%20Detection%20and%20Classification/SavedRuns/ClothingDetection/detectron2_maskrcnn/fast_rcnn_accuracy.png)
![](https://github.com/cosminbvb/Clothing-Person-Detection/blob/main/Clothing%20Detection%20and%20Classification/SavedRuns/ClothingDetection/detectron2_maskrcnn/mask_rcnn_accuracy.png)
![](https://github.com/cosminbvb/Clothing-Person-Detection/blob/main/Clothing%20Detection%20and%20Classification/SavedRuns/ClothingDetection/detectron2_maskrcnn/loss_mask.png)
![](https://github.com/cosminbvb/Clothing-Person-Detection/blob/main/Clothing%20Detection%20and%20Classification/SavedRuns/ClothingDetection/detectron2_maskrcnn/loss_cls.png)
<a name="person-detection"/>

## Person Detection

### Dataset
The dataset I used for this task is the Penn-Fudan Database for Pedestrian Detection and Segmentation, which you can read more about [here](https://www.cis.upenn.edu/~jshi/ped_html/). The dataset is very small and better performance could be achived using broader datasets, but the main focus of the project was the clothing detection task.


### Metrics:
![](https://github.com/cosminbvb/Clothing-Recognition/blob/main/Clothing%20Detection%20and%20Classification/SavedRuns/PersonDetection/pytorch_faster_rcnn/ap.png)
![](https://github.com/cosminbvb/Clothing-Recognition/blob/main/Clothing%20Detection%20and%20Classification/SavedRuns/PersonDetection/pytorch_faster_rcnn/loss_lr.png)
