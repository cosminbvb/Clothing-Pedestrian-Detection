# Clothing & Person Detection

This repo focuses on the Clothing Detection task. However, I also chose to quickly train a Person Detection model (with a Pedestrian Detection dataset) and when performing inference pass each detected person to the Clothing Detection model. But, depending on the data used for inference, better speed/performance could be achived by directly passing the original images to the Clothing Detection model.

![](https://github.com/cosminbvb/Clothing-Pedestrian-Detection/blob/main/demo.gif)

* [Run with or without Docker](#run)
* Clothing Detection
  * [Dataset](#ds1)
  * [YOLOv5](#ap1)
  * [Mask RCNN with Detectron2](#ap2)
  * [Faster RCNN with PyTorch](#ap3)
* [Person Detection](#person-detection)


<a name="run"/>

## Run with or without Docker

### Running the flask app with Docker (recommended): 

Requirements:
- GNU/Linux x86_64 with kernel version > 3.10
- Docker >= 19.03
- docker-compose >= 1.28.0
- NVIDIA GPU with Architecture >= Kepler (or compute capability 3.0)
- NVIDIA Linux drivers >= 418.81.07 (Note that older driver releases or branches are unsupported.)
- [NVIDIA Container Toolkit and nvidia-docker2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

Then, just run ``` sudo docker-compose up -d --build ``` and go to ```127.0.0.1:5000``` in your browser.
To stop it, run ``` sudo docker-compose down ```.

### Running the flask app without Docker:

It is highly recommended to create a virtual environment and install these [packages](https://github.com/ultralytics/yolov5/blob/master/requirements.txt). 
Then, change directory to App, run the following commands:
```
export FLASK_APP=app
flask run
```
and go to ```127.0.0.1:5000``` in your browser.


<a name="clothing-recognition"/>

## Clothing Detection

***Disclaimer*** - This is not a fair comparison between the 3 approaches. Since the dataset is really big, a lot of time and resources 
would be required to reach convergence and find the best hyperparameters on each one of them.

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
Secondly, the labels need to be converted into YOLO format. You can do so with [this piece of code](https://github.com/cosminbvb/Clothing-Pedestrian-Detection/blob/main/Clothing%20Detection%20and%20Classification/deepfashion2_to_yolo.py).

Make sure to organise the directories in such way that the train and validation directories will each contain 2 directories called images and labels.

Then, you will have to choose a model to start training from. In my case, I went with the YOLOv5x configuration. 
As for the hyperparameters, they can be found [here](https://github.com/cosminbvb/Clothing-Pedestrian-Detection/blob/main/Clothing%20Detection%20and%20Classification/SavedRuns/ClothingDetection/yolov5/hyp.yaml).

Finally, I started training for 100 epochs with a batch size of 64 using the following command:
```
--img 640 --batch 64 --epochs 100 --data deepfashion2.yaml --weights yolov5x.pt
```


Now, let's take a look at how the model performed:

Results summary:

![Results](https://github.com/cosminbvb/Clothing-Pedestrian-Detection/blob/main/Clothing%20Detection%20and%20Classification/SavedRuns/ClothingDetection/yolov5/results.png?raw=true)
Precision is defined as the number of true positives over the number of true positives plus the number of false positives.
Recall is defined as the number of true positives over the number of true positives plus the number of false negatives.

Confusion matrix:
![Conf matrix](https://github.com/cosminbvb/Clothing-Pedestrian-Detection/blob/main/Clothing%20Detection%20and%20Classification/SavedRuns/ClothingDetection/yolov5/confusion_matrix.png?raw=true)

Precision-Recall curve:
The precision-recall curve shows the tradeoff between precision and recall. A high area under the curve represents both high recall and high precision, where high precision relates to a low false positive rate, and high recall relates to a low false negative rate. High scores for both show that the classifier is returning accurate results (high precision), as well as returning a majority of all positive results (high recall).
![PR curve](https://github.com/cosminbvb/Clothing-Pedestrian-Detection/blob/main/Clothing%20Detection%20and%20Classification/SavedRuns/ClothingDetection/yolov5/PR_curve.png?raw=true)

F1 curve:
Simply put, the F1 score combines precision and recall into one metric by calculating the harmonic mean between those two.
![F1 curve](https://github.com/cosminbvb/Clothing-Pedestrian-Detection/blob/main/Clothing%20Detection%20and%20Classification/SavedRuns/ClothingDetection/yolov5/F1_curve.png?raw=true)

The saved model can be found [here](https://github.com/cosminbvb/Clothing-Pedestrian-Detection/tree/main/Clothing%20Detection%20and%20Classification/SavedRuns/ClothingDetection/yolov5/weights).

<a name="ap2"/>

### Semantic Segmentation (Mask RCNN) with Detectron2

First of all, in order to train on a custom dataset, we'll need to register the dataset. The easiest way to do so, in my opinion, is to [convert the dataset into **coco** format](https://github.com/cosminbvb/Clothing-Pedestrian-Detection/blob/main/Clothing%20Detection%20and%20Classification/deepfashion2_to_coco.py) and then just run the ```register_coco_instances``` function they provide, passing the json and images directory paths as arguments.

Next, we need to define the configuration. Here, I chose the mask_rcnn_R_101_FPN_3x model configuration, 16 images per batch, 0.001 learning rate and 30k max interations.

Here is how the model performed while training:

![](https://github.com/cosminbvb/Clothing-Pedestrian-Detection/blob/main/Clothing%20Detection%20and%20Classification/SavedRuns/ClothingDetection/detectron2_maskrcnn/fast_rcnn_accuracy.png)
![](https://github.com/cosminbvb/Clothing-Pedestrian-Detection/blob/main/Clothing%20Detection%20and%20Classification/SavedRuns/ClothingDetection/detectron2_maskrcnn/mask_rcnn_accuracy.png)
![](https://github.com/cosminbvb/Clothing-Pedestrian-Detection/blob/main/Clothing%20Detection%20and%20Classification/SavedRuns/ClothingDetection/detectron2_maskrcnn/loss_mask.png)
![](https://github.com/cosminbvb/Clothing-Pedestrian-Detection/blob/main/Clothing%20Detection%20and%20Classification/SavedRuns/ClothingDetection/detectron2_maskrcnn/loss_cls.png)

Even though the above metrics look promising, while evaluating, we get the following results:

Bbox Evaluation:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.602
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.746
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.700
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.455
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.429
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.604
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.798
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.808
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.808
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.575
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.653
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.810
```

Segmentation Evaluation:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.578
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.740
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.682
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.412
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.290
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.583
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.752
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.762
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.762
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.575
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.637
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.763
```

Keep in mind that training to this point took about 30 hours on a 32GB VRAM GPU. Although these results are not exactly what we were hoping for, we need to once again remind ourselves how big and complex the dataset is. I am fairly convinced that with proper hardware and quite a bit of patience at the training stage, the model would actually perform really well.   

<a name="ap3"/>

### Faster RCNN with PyTorch

First, we have to define the dataset by creating a custom class inheriting the abstract class ```torch.utils.data.Dataset``` class and overriding the following methods:
- ```__len__``` returning the size of the dataset
- ```__getitem__``` returning the image object and the annotations dictionary, given the image index  

No data augmentation has been performed, since most of the images are already "manually" augmented, and the dataset is complex enough.

The next step after instantiating out custom dataset class is to create a ```DataLoader``` object. Here, try setting the batch_size as high as possible. Then, I defined a SGD optimizer, passing the learning rate as parameter.

We want to start from a model pre-trained on COCO and finetune it for our particular classes and so I chose the Faster RCNN ResNet50 FPN model. 

Finally, we can train for how many epochs we want to, by using the ```train_one_epoch``` and ```evaluate``` methods provided in ```tools/eninge.py```. With a few extra lines of code, we are also able to log the loss function, learning rate and the COCO evaluation metrics.  

When performing inference, we should not only set an adequate threshold, but also use Non-Max Suppression, and torchvision provides a plug-and-play function to perform this type of filtering. NMS is well explained [here](https://www.analyticsvidhya.com/blog/2020/08/selecting-the-right-bounding-box-using-non-max-suppression-with-implementation/).

Due to the lack of time and computing resources, this approach has only been trained for 6 epochs and only achieved an AP(IoU=0.50) of 0.70 and an AR(IoU=0.50:0.95) of 0.723, but could reach much better performance given a fair amount of time and resources.

<a name="person-detection"/>

## Person Detection

### Dataset
The dataset I used for this task is the Penn-Fudan Database for Pedestrian Detection and Segmentation, which you can read more about [here](https://www.cis.upenn.edu/~jshi/ped_html/). The dataset is very small and better performance could be achived using broader datasets, but the main focus of the project was the clothing detection task.

For this task I finetuned a Faster RCNN ResNet50 FPN model pretrained on COCO, using PyTorch. The steps followed are similar to the ones listed at the [Clothing Detection Faster RCNN with Pytorch](#ap3), with a few exceptions:
- Performed a tiny bit of data augmentation: RandomHorizontalFlip(0.5) for train images
- Used a LR Scheduler: StepLR  

### Metrics
![](https://github.com/cosminbvb/Clothing-Pedestrian-Detection/blob/main/Clothing%20Detection%20and%20Classification/SavedRuns/PersonDetection/pytorch_faster_rcnn/ap.png)
![](https://github.com/cosminbvb/Clothing-Pedestrian-Detection/blob/main/Clothing%20Detection%20and%20Classification/SavedRuns/PersonDetection/pytorch_faster_rcnn/loss_lr.png)
