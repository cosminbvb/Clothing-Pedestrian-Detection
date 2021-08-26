import os
import numpy as np
import torch
from PIL import Image
import cv2

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.tensorboard import SummaryWriter

from tools.engine import train_one_epoch, evaluate
import tools.utils as utils
import tools.transforms as T

import matplotlib.pyplot as plt
import matplotlib.patches as patches


class PennFudanDataset(object):  
    # Obs: this dataset also provides masks
    #      dataset isn't devided
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
        # print(self.imgs[0])  # FudanPed00001.png

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # print(img.shape) # torch.Size([3, 351, 309])
        return img, target  # img = tensor

    def __len__(self):
        return len(self.imgs)


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_dataset(dataset_name):
    """
        Returns train & validation datasets given the dataset name 
    """
    if dataset_name == "PennFudanPed":
        train_path = valid_path = 'Datasets/PennFudanPed'
        dataset_train = PennFudanDataset(train_path, get_transform(train=True))
        dataset_valid = PennFudanDataset(valid_path, get_transform(train=False))
        # split the dataset
        indices = torch.randperm(len(dataset_train)).tolist()
        dataset_train = torch.utils.data.Subset(dataset_train, indices[:-20])
        dataset_valid = torch.utils.data.Subset(dataset_valid, indices[-20:])
    else:
        print('Unknown dataset')
        return None, None

    return dataset_train, dataset_valid


def get_dataLoaders(dataset_train, dataset_valid):
    """
        Returns dataLoader objects for train and validation datasets 
    """
    # define training and validation data loaders
    dataLoader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)  # set batch_size as high as possible
    
    dataLoader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    
    return dataLoader_train, dataLoader_valid 


def inference(model_path, dataset_valid):
    # shows test images with the predicted bboxes and coresponding score
    # also uses non maximum supression to remove some redundant boxes

    device = torch.device('cuda')
    model = torch.load(model_path)

    for i in range(len(dataset_valid)):
        image = dataset_valid[i][0]
        print(image)
        print(type(image))
        # put the model in evaluation mode
        model.eval()
        with torch.no_grad():
            prediction = model([image.to(device)])

        boxes = prediction[0]["boxes"]
        scores = prediction[0]["scores"]

        keep = torchvision.ops.nms(boxes, scores, 0.2)  # indexes of all the bounding boxes we should keep

        boxes = boxes.tolist()
        scores = scores.tolist()

        threshold = 0.6

        image = image.permute(1,2,0)
        image = image.numpy()

        fig, ax = plt.subplots(1)
        ax.imshow(image)
        for i in keep:
            if scores[i] < threshold:
                continue
            x1, y1, x2, y2 = map(int, boxes[i])
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1,
                            edgecolor='r', facecolor="none")
            ax.text(x1, y1, str(round(scores[i], 2)*100) + "%", color="black", bbox={'facecolor': 'lime', 'alpha': 0.5})
            ax.add_patch(rect)
        plt.show()


def train(dataLoader_train, dataLoader_valid):
    # train on the GPU or on the CPU, if a GPU is not available
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # only train on gpu
    device = torch.device('cuda')
    
    # our dataset has two classes only - background and person
    num_classes = 2

    # pre-trained
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005)

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=25,
                                                   gamma=0.8)
    # Decays the learning rate of each parameter group by gamma every step_size epochs.

    num_epochs = 300

    log_writer = SummaryWriter()  # tensorboard log writer
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        logs = train_one_epoch(model, optimizer, dataLoader_train, device, epoch, print_freq=100)
        
        # update the learning rate
        # lr_scheduler.step()

        # evaluate on the validation dataset and get the stats 
        evaluator = evaluate(model, dataLoader_valid, device=device)
        for iou_type, coco_eval in evaluator.coco_eval.items():
            if iou_type == 'bbox':
                metrics = coco_eval.stats

        log_writer.add_scalar("loss", logs.loss.value, epoch)  # logging the loss function
        log_writer.add_scalar("lr", logs.lr.value, epoch)  # and the learning rate
        log_writer.add_scalar("AP(IoU=0.50:0.95 | area=   all | maxDets=100)", metrics[0].round(3), epoch)
        log_writer.add_scalar("AP(IoU=0.50      | area=   all | maxDets=100)", metrics[1].round(3), epoch)
        log_writer.add_scalar("AP(IoU=0.75      | area=   all | maxDets=100)", metrics[2].round(3), epoch)
        log_writer.add_scalar("AR(IoU=0.50:0.95 | area=   all | maxDets=  1)", metrics[6].round(3), epoch)
        log_writer.add_scalar("AR(IoU=0.50:0.95 | area=   all | maxDets= 10)", metrics[7].round(3), epoch)
        log_writer.add_scalar("AR(IoU=0.50:0.95 | area=   all | maxDets=100)", metrics[8].round(3), epoch)

        # save every 50 epochs
        if epoch % 50 == 0 and epoch != 0:
            torch.save(model, f"SavedRuns/PersonDetection/pytorch_faster_rcnn/{epoch}_4.pt")


    log_writer.flush()
    log_writer.close()

    torch.save(model, "SavedRuns/PersonDetection/pytorch_faster_rcnn/trained_4.pt")


def eval(model_path, dataLoader_valid):
    device = torch.device('cuda')
    model = torch.load(model_path)
    model.to(device)
    evaluate(model, dataLoader_valid, device=device)


if __name__ == "__main__":

    # # Mask example:
    # mask = Image.open('PennFudanPed/PedMasks/FudanPed00001_mask.png')
    # # each mask instance has a different color, from zero to N, where
    # # N is the number of instances. In order to make visualization easier,
    # # let's adda color palette to the mask.
    # mask.putpalette([
    #     0, 0, 0, # black background
    #     255, 0, 0, # index 1 is red
    #     255, 255, 0, # index 2 is yellow
    #     255, 153, 0, # index 3 is orange
    # ])
    # mask.show()


    # get the dataset and dataLoaders:
    dataset_train, dataset_valid = get_dataset("PennFudanPed")
    dataLoader_train, dataLoader_valid = get_dataLoaders(dataset_train, dataset_valid)

    # train (and eval):
    # train(dataLoader_train, dataLoader_valid)

    # saved model path
    model_path = 'SavedRuns/PersonDetection/pytorch_faster_rcnn/50_3.pt'

    # only eval:
    eval(model_path, dataLoader_valid)

    # inference:
    # inference(model_path, dataset_valid)
