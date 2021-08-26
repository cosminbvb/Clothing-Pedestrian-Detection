import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO
import numpy as np

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.tensorboard import SummaryWriter

from tools.engine import train_one_epoch, evaluate
import tools.utils as utils
import tools.transforms as T

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class DeepFashion2(torch.utils.data.Dataset):
    
    classes = ['-', 'short sleeve top',
    'long sleeve top', 'short sleeve outwear',
    'long sleeve outwear', 'vest', 'sling', 'shorts',
    'trousers', 'skirt', 'short sleeve dress',
    'long sleeve dress', 'vest dress', 'sling dress']

    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = np.array(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        coco_annotation = self.coco.loadAnns(ann_ids)
        path = self.coco.loadImgs([img_id])[0]['file_name']

        # open the input image
        img = Image.open(os.path.join(self.root, path))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        areas = []
        # masks = []
        iscrowd = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(coco_annotation[i]['category_id'])
            areas.append(coco_annotation[i]['area'])
            # masks.append(coco_annotation[i]['segmentation'])
            # print(coco_annotation[i]['segmentation'])
            # print()
            iscrowd.append(coco_annotation[i]['iscrowd'])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # masks = torch.as_tensor(masks, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        img_id = torch.tensor([img_id])

        anno_dict = {} 
        anno_dict["boxes"] = boxes
        anno_dict["labels"] = labels
        # anno_dict["masks"] = masks
        anno_dict["image_id"] = img_id
        anno_dict["area"] = areas
        anno_dict["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, anno_dict

    def __len__(self):
        return len(self.ids)


# just added ToTensor
# training set already has some data augmentation
def get_transform():
    custom_transforms = [torchvision.transforms.ToTensor()]
    return torchvision.transforms.Compose(custom_transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


def train():
    device = torch.device('cuda')
    
    num_classes = 14  # 13 for clothing, 1 for background

    # start from pretrained weights or from a checkpoint:

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # model = torch.load("SavedModels/trained.pt")

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(device)

    log_writer = SummaryWriter()  # tensorboard log writer

    train_dir = 'Datasets/Deepfashion2/train/image'
    train_json = 'Datasets/Deepfashion2/train.json'

    dataset = DeepFashion2(root=train_dir,
                        annotation=train_json,
                        transforms=get_transform())
    
    # set batch_size as high as possible
    data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=4,
    collate_fn=collate_fn)


    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001)
    
    num_epochs = 10
    for epoch in range(num_epochs):
        logs = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
        
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

            
    log_writer.flush()
    log_writer.close()

    torch.save(model, "SavedRuns/ClothingDetection/pytorch_faster_rcnn/trained.pt")


def inference(model_path):
    # inference on the validation dataset
    device = torch.device('cuda')
    model = torch.load(model_path)
    model.to(device)

    valid_dir = 'Datasets/Deepfashion2/validation/image'
    valid_json = 'Datasets/Deepfashion2/valid.json'
    dataset = DeepFashion2(root=valid_dir,
                           annotation=valid_json,
                           transforms=get_transform())

    for i in range(len(dataset)):
        image = dataset[i][0]
        # put the model in evaluation mode
        model.eval()
        with torch.no_grad():
            prediction = model([image.to(device)])

        boxes = prediction[0]["boxes"]
        scores = prediction[0]["scores"]
        labels = prediction[0]["labels"]

        keep = torchvision.ops.nms(boxes, scores, 0.3)  # indexes of all the bounding boxes we should keep

        boxes = boxes.tolist()
        scores = scores.tolist()

        image = image.permute(1,2,0)
        image = image.numpy()

        fig, ax = plt.subplots(1)
        ax.imshow(image)
        for i in keep:
            if scores[i] < 0.50:
                continue
            x1, y1, x2, y2 = map(int, boxes[i])
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1,
                            edgecolor='r', facecolor="none")
            ax.text(x1, y1, DeepFashion2.classes[labels[i].item()] + " " + str(round(scores[i]*100, 2)) + "%", color='lime')
            ax.add_patch(rect)
        plt.show()


def load_and_eval(model_path):
    device = torch.device('cuda')
    model = torch.load(model_path)

    valid_dir = 'Datasets/Deepfashion2/validation/image'
    valid_json = 'Datasets/Deepfashion2/valid.json'
    dataset = DeepFashion2(root=valid_dir,
                           annotation=valid_json,
                           transforms=get_transform())
    dataset = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=collate_fn)
    
    # move model to the right device
    model.to(device)

    # evaluate 
    evaluate(model, dataset, device=device)
    print('-------------------------------------')


if __name__ == "__main__":
    # train()
    # load_and_eval('SavedRuns/ClothingDetection/pytorch_faster_rcnn/trained.pt')
    inference('SavedRuns/ClothingDetection/pytorch_faster_rcnn/trained.pt')