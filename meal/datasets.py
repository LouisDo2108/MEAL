import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
import albumentations as A

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, default_collate
import timm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.augmentations import letterbox


def get_cls_model():
    
    def overwrite_key(state_dict):
        for key in list(state_dict.keys()):
            state_dict[key.replace("model.", "")] = state_dict.pop(key)
        return state_dict
    
    model = timm.create_model("tf_efficientnet_b0", num_classes=4)
    checkpoint = torch.load(
        str(ROOT / "checkpoints" / "cls_model.pth"), map_location=torch.device('cpu')
    )["model"]
    model.load_state_dict(overwrite_key(checkpoint))
    model.cuda()
    return model


def get_backbone_model():
    dnn = False
    half = False
    device = ""
    device = select_device(device)
    model = DetectMultiBackend(
        weights=str(ROOT / "checkpoints" / "backbone.pt"),
        device=device,
        dnn=dnn,
        data=None,
        fp16=half,
    )
    imgsz = [480, 480]
    model.warmup(imgsz=(1 if model.pt or model.triton else 4, 3, *imgsz))
    return model

    
def collate_fn(batch):
    small_list, medium_list, large_list, bbox_list, global_embedding_list, cls_list = [], [], [], [], [], []
    for x, y in batch:
        small_list.append(x[0])
        medium_list.append(x[1])
        large_list.append(x[2])
        _box = torch.tensor(x[3])
        _box[:, 2], _box[:, 3] = _box[:, 0] + _box[:, 2], _box[:, 1] + _box[:, 3]
        bbox_list.append(_box)
        global_embedding_list.append(x[4])
        cls_list.append(y)
        
    # Collate tensors within each list
    small_list = default_collate(small_list)
    medium_list = default_collate(medium_list)
    large_list = default_collate(large_list)
    global_embedding_list = default_collate(global_embedding_list)
    cls_list = default_collate(cls_list)
    return (
        small_list,
        medium_list,
        large_list,
        bbox_list,
        global_embedding_list,
    ), cls_list


def get_dataloaders(train_transforms, val_transforms, batch_size, fold=0):
    train_ds = VofoDataset("./vocal-folds", train_val="train",
                                 transform=train_transforms, fold=fold)
    train_dl = DataLoader(train_ds, batch_size=batch_size, drop_last=False, collate_fn=collate_fn, shuffle=True)

    val_ds = VofoDataset("./vocal-folds", train_val="val", fold=fold,
                               transform=val_transforms)
    val_dl = DataLoader(val_ds, batch_size=batch_size, drop_last=False, collate_fn=collate_fn, shuffle=True)

    return train_dl, val_dl


def get_transforms():
    train_transforms = A.Compose(
        [
            A.augmentations.crops.transforms.RandomSizedBBoxSafeCrop(360, 480, p=0.3),
            A.augmentations.geometric.transforms.Affine(scale=0.5, translate_percent=0.1, p=0.3),
            A.augmentations.transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.7, hue=0.015, p=0.3)
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["bbox_categories"]),
    )
    val_transforms = None
    return train_transforms, val_transforms


class VofoDataset(Dataset):	
    	
    def __init__(self, root_dir, train_val, fold=0, transform=None, target_transform=None):	
        super().__init__()	
        self.root_dir = Path("./vocal-folds/")	
        self.train_val = train_val	
        self.fold = fold	
        self.transform = transform	
        self.target_transform = target_transform	
        self.img_dir = self.root_dir / "images" / "Train_4classes"	
        self.label_dict = {	
            "Nor-VF": 0,	
            "Non-VF": 1,	
            "Ben-VF": 2,	
            "Mag-VF": 3,	
        }	
        self.backbone_model = get_backbone_model()	
        self.cls_model = get_cls_model()	
        self.backbone_model.eval()	
        self.cls_model.eval()	
        self.backbone_transform = transforms.ToTensor()	
        self.cls_transform = transforms.Compose([	
            transforms.ToTensor(),	
            transforms.Resize((256, 256)),	
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),	
        ])	
        with open(self.root_dir / "annotations" / f"annotation_{fold}_{train_val}.json", "r") as f:	
            js = json.load(f)	
            	
        self.data = []	
        self.image_id_to_index = {}	
        self.categories = js["categories"]	
        for i, image in enumerate(js["images"]):	
            new_dict = {}	
            # Extract the class id from the orignal image name	
            cls = image["file_name"].split("_")[0]	
            # Get the image name in the image folder	
            filename = image["file_name"].split("_")[-1]	
            new_dict["id"] = image["id"]	
            new_dict["file_name"] = filename	
            new_dict["img_path"] = os.path.join(self.img_dir, filename)	
            new_dict["bbox"] = []	
            new_dict["bbox_categories"] = []	
            new_dict["cls"] = self.label_dict[cls]	
            self.data.append(new_dict)	
            self.image_id_to_index[image["id"]] = i
        for annot in js["annotations"]:	
            img_index = self.image_id_to_index[annot["image_id"]]	
            img_data = self.data[img_index]	
            img_data["bbox"].append(annot["bbox"])	
            img_data["bbox_categories"].append(annot["category_id"])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Retrieve data from the dataset at the given index
        data = self.data[idx]

        # Extract image path, bounding boxes, bbox categories, and class ID
        img_path = data["img_path"]
        bboxes = data.get("bbox", [])
        bbox_categories = data.get("bbox_categories", [])
        cls = data["cls"]

        # Load image from path
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # If no bounding boxes are found, add a default bounding box
        if not bboxes:
            bboxes.append([0, 0, img.shape[1], img.shape[0]])
            bbox_categories.append(7)

        # Apply image transforms if available
        if self.transform:
            transformed = self.transform(image=img, bboxes=bboxes, bbox_categories=bbox_categories)
            img = transformed["image"]
            bboxes = transformed["bboxes"]
            bbox_categories = transformed["bbox_categories"]

        # Apply target transforms if available
        if self.target_transform:
            transformed = self.target_transform(image=img, bboxes=bboxes, bbox_categories=bbox_categories)
            img = transformed["image"]
            bboxes = transformed["bboxes"]
            bbox_categories = transformed["bbox_categories"]
            
        # Convert image to YOLO format and extract YOLO features and bounding boxes
        x = letterbox(img.copy(), 480, auto=False, stride=32)[0]
        x = x.transpose((2, 0, 1))
        x = torch.from_numpy(x).unsqueeze(0).cuda().float().div(255.0)
        with torch.no_grad():
             small, medium, large, pred = self.backbone_model(x, feature_map=True)

        # Convert image to classification format and extract classification features
        x = self.cls_transform(img.copy()).unsqueeze(0).cuda()
        with torch.no_grad():
            global_embedding = self.cls_model.forward_features(x)
    
        return (small[0], medium[0], large[0], bboxes, global_embedding[0]), cls
