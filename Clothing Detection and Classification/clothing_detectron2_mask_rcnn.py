import os
import copy
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

import detectron2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.data import build_detection_test_loader

setup_logger()

# register dataset
register_coco_instances('DeepFashion2_train', {},
                        'Datasets/Deepfashion2/train.json',
                        'Datasets/Deepfashion2/train/image')
register_coco_instances('DeepFashion2_valid', {},
                        'Datasets/Deepfashion2/valid.json',
                        'Datasets/Deepfashion2/validation/image')

# # visualize validation data (loads faster since it s smaller)
# my_dataset_val_metadata = MetadataCatalog.get("DeepFashion2_valid")
# dataset_dicts = DatasetCatalog.get("DeepFashion2_valid")

# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_val_metadata, scale=0.5)
#     vis = visualizer.draw_dataset_dict(d)
#     plt.figure(figsize = (14, 10))
#     plt.imshow(cv2.cvtColor(vis.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
#     plt.show()


# config

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("DeepFashion2_train",)
cfg.DATASETS.TEST = ("DeepFashion2_valid",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 16
cfg.SOLVER.BASE_LR = 0.001  # pick a good LR
cfg.SOLVER.MAX_ITER = 30000    
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13  # (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.OUTPUT_DIR = 'SavedRuns'


# train 

# trainer = DefaultTrainer(cfg) 
# trainer.resume_or_load(resume=False)
# trainer.train()


# inference on 100 random images

cfg.MODEL.WEIGHTS = 'SavedRuns/ClothingDetection/detectron2_maskrcnn/model_final.pth'  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6   # set a custom testing threshold
predictor = DefaultPredictor(cfg)
my_dataset_val_metadata = MetadataCatalog.get("DeepFashion2_valid")
dataset_dicts = DatasetCatalog.get("DeepFashion2_valid")
for d in random.sample(dataset_dicts, 100):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=my_dataset_val_metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    plt.show()


# eval

# model = build_model(cfg)
# DetectionCheckpointer(model).load('SavedRuns/ClothingDetection/detectron2_maskrcnn/model_final.pth')

# evaluator = COCOEvaluator("DeepFashion2_valid")
# val_loader = build_detection_test_loader(cfg, "DeepFashion2_valid")
# print(inference_on_dataset(model, val_loader, evaluator))