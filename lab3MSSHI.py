import os
import sys
import json
import shutil
import random
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import transforms
from torchvision.ops import box_convert

from PIL import Image
import cv2
from tqdm import tqdm

from kaggle import api as kaggle_api
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# -------------------
# CONFIG (edit here)
# -------------------
CONFIG = {
    # Dataset
    "dataset_dir": "",  # set via CLI --dataset_dir or here; should contain images/ and annotations/
    "work_dir": "work_tomato",
    "outputs_dir": "outputs",
    "classes": ["tomato"],

    # YOLOv8
    "yolo": {
        "weights": "yolov8n.pt",
        "epochs": 30,
        "imgsz": 640,
        "batch": 16,
        "eval_conf": 0.001  # low conf for COCO eval to avoid empty detections
    },

    # TorchVision common
    "tv": {
        "epochs": 10,
        "batch_train": 4,
        "batch_val": 2,
        "lr": 5e-4,
        "weight_decay": 1e-4
    },

    # Faster R-CNN
    "frcnn": {
        "weights": "DEFAULT",  # torchvision preset
        "label_offset": 0,     # tomato -> 1 (1-based labels for FG), background handled internally
        "cat_id_offset": 0     # predictions are already 1-based for COCO export
    },

    # RetinaNet
    "retinanet": {
        "weights": None,         # detection head random init, backbone pretrained below
        "weights_backbone": "DEFAULT",
        "num_classes": 1,        # FG classes; labels must be 0..num_classes-1
        "label_offset": -1,      # tomato -> 0 (0-based labels for FG)
        "cat_id_offset": 1,      # map 0-based predictions to COCO category_id=1 for eval
        "eval_conf": 0.0         # ensure non-empty eval results early in training
    },

    # COCO evaluation
    "coco": {
        "maxDets": [1, 10, 100]  # required for summarize() and AR@100 at stats[8]
    }
}

# -------------------
# Helpers and setup
# -------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".JPG", ".PNG"}

def device_info():
    use_cuda = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    print(f"CUDA available: {use_cuda} | Device: {name}")
    return "cuda:0" if use_cuda else "cpu"

def resolve_tomato_root(base: Path) -> Path:
    base = base.resolve()
    def has_images(p: Path) -> bool:
        try:
            return any(f.suffix.lower() in IMG_EXTS for f in p.rglob("*"))
        except Exception:
            return False
    if base.exists() and base.is_dir() and has_images(base):
        return base
    if base.name.lower() == "annotations" and base.parent.exists():
        par = base.parent
        if has_images(par):
            return par
    if base.parent.exists():
        for cand in [base] + [p for p in base.parent.iterdir() if p.is_dir()]:
            img_dir = cand / "images"
            if img_dir.exists() and img_dir.is_dir() and has_images(img_dir):
                return cand
    if base.parent.exists() and has_images(base.parent):
        return base.parent
    return base

def download_variant10_dataset(root: Path) -> Path:
    ds_root = root / "datasets"
    ds_root.mkdir(parents=True, exist_ok=True)
    slug = "andrewmvd/tomato-detection"
    print(f"Downloading Kaggle dataset: {slug} -> {ds_root}")
    kaggle_api.dataset_download_files(dataset=slug, path=str(ds_root), unzip=True)
    return resolve_tomato_root(ds_root)

def voc_parse_xml(xml_path: Path):
    import xml.etree.ElementTree as ET
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    objs = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        bnd = obj.find("bndbox")
        xmin = int(float(bnd.find("xmin").text))
        ymin = int(float(bnd.find("ymin").text))
        xmax = int(float(bnd.find("xmax").text))
        ymax = int(float(bnd.find("ymax").text))
        objs.append((name, xmin, ymin, xmax, ymax, w, h))
    return objs, w, h

def voc_bbox_to_yolo(xmin, ymin, xmax, ymax, iw, ih):
    x_c = (xmin + xmax) / 2.0 / iw
    y_c = (ymin + ymax) / 2.0 / ih
    bw = (xmax - xmin) / iw
    bh = (ymax - ymin) / ih
    return x_c, y_c, bw, bh

def prepare_yolo_and_coco(raw_root: Path, work_root: Path, classes: List[str]):
    if work_root.exists():
        shutil.rmtree(work_root)
    for sp in ["train", "val", "test"]:
        (work_root / "images" / sp).mkdir(parents=True, exist_ok=True)
        (work_root / "labels" / sp).mkdir(parents=True, exist_ok=True)
    images = [p for p in raw_root.rglob("*") if p.suffix in IMG_EXTS]
    xmls = {p.stem: p for p in raw_root.rglob("*.xml")}
    images = list({p.resolve() for p in images})
    random.seed(42)
    random.shuffle(images)
    n = len(images)
    n_train = max(1, int(0.7 * n))
    n_val = max(1, int(0.2 * n))
    n_test = max(1, n - n_train - n_val)
    splits = {"train": images[:n_train], "val": images[n_train:n_train + n_val], "test": images[n_train + n_val:]}
    print({k: len(v) for k, v in splits.items()})
    if n == 0:
        raise RuntimeError(f"No images found under {raw_root}. Ensure --dataset_dir points to a parent folder containing images and annotations.")
    class_to_id = {c: i for i, c in enumerate(classes)}
    for sp, paths in splits.items():
        for img_path in tqdm(paths, desc=f"Copy+Label {sp}"):
            dst_img = work_root / "images" / sp / img_path.name
            shutil.copy(img_path, dst_img)
            lbl_path = work_root / "labels" / sp / f"{img_path.stem}.txt"
            lbl_path.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            native = img_path.with_suffix(".txt")
            if native.exists():
                lines = [ln.strip() for ln in native.read_text().splitlines()]
            else:
                if img_path.stem in xmls:
                    objs, iw, ih = voc_parse_xml(xmls[img_path.stem])
                    for name, xmin, ymin, xmax, ymax, iw, ih in objs:
                        cid = class_to_id.get(name, 0)
                        x, y, w, h = voc_bbox_to_yolo(xmin, ymin, xmax, ymax, iw, ih)
                        lines.append(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            with open(lbl_path, "w") as f:
                f.write("\n".join(lines))
    data_yaml = work_root / "data.yaml"
    data_yaml.write_text(
        f"train: {str((work_root/'images'/'train').resolve())}\n"
        f"val: {str((work_root/'images'/'val').resolve())}\n"
        f"test: {str((work_root/'images'/'test').resolve())}\n\n"
        f"nc: {len(classes)}\n"
        f"names: {classes}\n"
    )
    print("Wrote", data_yaml)
    def yolo_to_coco(split: str) -> Dict:
        from PIL import Image as PILImage
        img_dir = work_root / "images" / split
        lbl_dir = work_root / "labels" / split
        imgs = [p for p in img_dir.glob("*") if p.suffix in IMG_EXTS]
        imgs.sort()
        categories = [{"id": i + 1, "name": c, "supercategory": "object"} for i, c in enumerate(classes)]
        images_json, annotations = [], []
        ann_id = 1
        for img_id, p in enumerate(imgs, 1):
            w, h = PILImage.open(p).size
            images_json.append({"id": img_id, "file_name": str(p), "width": w, "height": h})
            lp = lbl_dir / f"{p.stem}.txt"
            if lp.exists():
                for ln in lp.read_text().splitlines():
                    if not ln.strip():
                        continue
                    cid_s, x, y, bw, bh = ln.strip().split()
                    cid = int(cid_s); x = float(x); y = float(y); bw = float(bw); bh = float(bh)
                    x0 = (x - bw / 2) * w; y0 = (y - bh / 2) * h
                    aw = bw * w; ah = bh * h
                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cid + 1,
                        "bbox": [x0, y0, aw, ah],
                        "area": aw * ah,
                        "iscrowd": 0,
                        "segmentation": []
                    })
                    ann_id += 1
        info = {"description": "Tomato Detection Variant 10", "url": "", "version": "1.0", "year": 2025, "contributor": "student", "date_created": datetime.now().isoformat()}
        licenses = []
        return {"info": info, "licenses": licenses, "images": images_json, "annotations": annotations, "categories": categories}
    for sp in ["train", "val", "test"]:
        coco = yolo_to_coco(sp)
        out_json = work_root / f"coco_{sp}.json"
        out_json.write_text(json.dumps(coco))
        print("Wrote", out_json)
    return data_yaml, work_root

def draw_yolo_boxes(img_path: Path, lbl_path: Path, names: List[str]):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if lbl_path.exists():
        for ln in lbl_path.read_text().splitlines():
            if not ln.strip():
                continue
            cid, x, y, bw, bh = ln.split()
            cid = int(cid); x = float(x); y = float(y); bw = float(bw); bh = float(bh)
            x0 = int((x - bw / 2) * w); y0 = int((y - bh / 2) * h)
            x1 = int((x + bw / 2) * w); y1 = int((y + bh / 2) * h)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(img, names[cid], (x0, max(0, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return img

# -------------------
# YOLO training + COCO eval (ID-aligned)
# -------------------
def yolo_manual_coco_eval(yolo_model, work_root: Path, device: str, conf_thr: float, maxDets: List[int]):
    gt_json = work_root / "coco_val.json"
    coco_gt = COCO(str(gt_json))
    img_id_map = {Path(im["file_name"]).name: im["id"] for im in coco_gt.dataset["images"]}
    val_dir = work_root / "images" / "val"
    val_imgs = sorted([p for p in val_dir.glob("*") if p.suffix.lower() in IMG_EXTS], key=lambda x: x.name)
    if len(val_imgs) == 0:
        raise RuntimeError(f"No val images in {val_dir}")
    res = yolo_model.predict([str(p) for p in val_imgs], imgsz=CONFIG["yolo"]["imgsz"], conf=conf_thr, device=0 if device.startswith("cuda") else "cpu", verbose=False)
    detections = []
    for p, r in zip(val_imgs, res):
        name = p.name
        image_id = img_id_map.get(name, None)
        if image_id is None:
            stem_map = {Path(im["file_name"]).stem: im["id"] for im in coco_gt.dataset["images"]}
            image_id = stem_map.get(p.stem, None)
        if image_id is None:
            continue
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        xywh = xyxy.copy()
        xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
        xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
        xywh[:, 0] = xyxy[:, 0]
        xywh[:, 1] = xyxy[:, 1]
        for b, s, c in zip(xywh, confs, clss):
            detections.append({
                "image_id": int(image_id),
                "category_id": int(c) + 1,
                "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                "score": float(s)
            })
    # Guard for empty
    if not detections:
        cat_ids = coco_gt.getCatIds()
        per_class = {coco_gt.loadCats([cid])[0]["name"]: 0.0 for cid in cat_ids}
        return {"mAP_50_95": 0.0, "mAP_50": 0.0, "AR100": 0.0, "AP_per_class": per_class}
    coco_dt = coco_gt.loadRes(detections)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.maxDets = maxDets
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    metrics = {"mAP_50_95": float(coco_eval.stats[0]), "mAP_50": float(coco_eval.stats[1]), "AR100": float(coco_eval.stats[8]), "AP_per_class": {}}
    for cid in coco_gt.getCatIds():
        ce = COCOeval(coco_gt, coco_dt, iouType='bbox')
        ce.params.catIds = [cid]
        ce.params.maxDets = maxDets
        ce.evaluate(); ce.accumulate(); ce.summarize()
        name = coco_gt.loadCats([cid])[0]["name"]
        metrics["AP_per_class"][name] = float(ce.stats[0])
    return metrics

def train_eval_yolo(data_yaml: Path, work_root: Path, device: str):
    yolo = YOLO(CONFIG["yolo"]["weights"])
    yolo.train(
        data=str(data_yaml),
        epochs=CONFIG["yolo"]["epochs"],
        imgsz=CONFIG["yolo"]["imgsz"],
        batch=CONFIG["yolo"]["batch"],
        project=str((work_root / "runs_yolo").resolve()),
        name="tomato_yolo",
        device=0 if device.startswith("cuda") else "cpu",
        verbose=True
    )
    # Internal val for plots only; COCO metrics via manual eval for ID alignment
    yolo.val(
        data=str(data_yaml),
        imgsz=CONFIG["yolo"]["imgsz"],
        save_json=False,
        project=str((work_root / "runs_yolo").resolve()),
        name="tomato_yolo_val",
        device=0 if device.startswith("cuda") else "cpu",
        verbose=True
    )
    metrics = yolo_manual_coco_eval(yolo, work_root, device, conf_thr=CONFIG["yolo"]["eval_conf"], maxDets=CONFIG["coco"]["maxDets"])
    return yolo, metrics

# -------------------
# TorchVision datasets/models
# -------------------
class YoloDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir: Path, lbl_dir: Path, classes: List[str], transforms=None, label_offset: int = 0):
        self.imgs = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in IMG_EXTS])
        self.lbl_dir = lbl_dir
        self.classes = classes
        self.transforms = transforms
        self.label_offset = label_offset  # 0 -> 1-based FG (FRCNN), -1 -> 0-based FG (RetinaNet)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        p = self.imgs[idx]
        img = Image.open(p).convert("RGB")
        w, h = img.size
        lbl_path = self.lbl_dir / f"{p.stem}.txt"
        boxes, labels = [], []
        if lbl_path.exists():
            for ln in lbl_path.read_text().splitlines():
                if not ln.strip():
                    continue
                cid, x, y, bw, bh = ln.split()
                cid = int(cid)
                cid = cid + 1 + self.label_offset  # tomato: 1 (offset=0), 0 (offset=-1)
                x, y, bw, bh = map(float, (x, y, bw, bh))
                x0 = (x - bw / 2) * w; y0 = (y - bh / 2) * h
                x1 = (x + bw / 2) * w; y1 = (y + bh / 2) * h
                boxes.append([x0, y0, x1, y1])
                labels.append(cid)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([idx])
        }
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target, str(p)

def collate_fn(batch):
    imgs, targets, paths = zip(*batch)
    return list(imgs), list(targets), list(paths)

def build_fasterrcnn(num_fg_classes: int):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=CONFIG["frcnn"]["weights"])
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_fg_classes + 1)
    return model

def build_retinanet(num_fg_classes: int):
    model = torchvision.models.detection.retinanet_resnet50_fpn(
        weights=CONFIG["retinanet"]["weights"],
        weights_backbone=CONFIG["retinanet"]["weights_backbone"],
        num_classes=num_fg_classes
    )
    return model

@torch.no_grad()
def evaluate_to_coco_json(model, dl, device: str, coco_gt_json: Path, score_thresh: float, category_id_offset: int, maxDets: List[int]) -> Tuple[Dict, List[Dict]]:
    model.eval()
    coco_gt = COCO(str(coco_gt_json))
    results = []
    img_id_map = {Path(im["file_name"]).name: im["id"] for im in coco_gt.dataset["images"]}
    for imgs, targets, paths in tqdm(dl, desc="Infer val"):
        imgs = [img.to(device) for img in imgs]
        outputs = model(imgs)
        for out, p in zip(outputs, paths):
            boxes = out["boxes"].detach().cpu().numpy()
            scores = out["scores"].detach().cpu().numpy()
            labels = out["labels"].detach().cpu().numpy()
            if boxes.shape[0] > 0:
                xywh = box_convert(torch.from_numpy(boxes), in_fmt="xyxy", out_fmt="xywh").numpy()
            else:
                xywh = np.zeros((0, 4), dtype=np.float32)
            img_name = Path(p).name
            image_id = img_id_map.get(img_name)
            if image_id is None:
                stem_map = {Path(im["file_name"]).stem: im["id"] for im in coco_gt.dataset["images"]}
                image_id = stem_map.get(Path(p).stem, None)
            if image_id is None:
                continue
            for b, s, lb in zip(xywh, scores, labels):
                if s < score_thresh:
                    continue
                results.append({
                    "image_id": int(image_id),
                    "category_id": int(lb) + category_id_offset,
                    "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                    "score": float(s)
                })
    return coco_gt, results

def coco_eval_from_results(coco_gt: COCO, detections: List[Dict], maxDets: List[int]):
    # Guard for empty results
    if not detections:
        cat_ids = coco_gt.getCatIds()
        per_class = {coco_gt.loadCats([cid])[0]["name"]: 0.0 for cid in cat_ids}
        return {"mAP_50_95": 0.0, "mAP_50": 0.0, "AR100": 0.0, "AP_per_class": per_class}
    coco_dt = coco_gt.loadRes(detections)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.maxDets = maxDets
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    metrics = {"mAP_50_95": float(coco_eval.stats[0]), "mAP_50": float(coco_eval.stats[1]), "AR100": float(coco_eval.stats[8])}
    per_class = {}
    for cid in coco_gt.getCatIds():
        ce = COCOeval(coco_gt, coco_dt, iouType='bbox')
        ce.params.catIds = [cid]
        ce.params.maxDets = maxDets
        ce.evaluate(); ce.accumulate(); ce.summarize()
        name = coco_gt.loadCats([cid])[0]["name"]
        per_class[name] = float(ce.stats[0])
    metrics["AP_per_class"] = per_class
    return metrics

def train_torchvision_detector(model_type: str, work_root: Path, device: str, classes: List[str], epochs: int, lr: float, wd: float):
    img_tr = work_root / "images" / "train"
    lbl_tr = work_root / "labels" / "train"
    img_val = work_root / "images" / "val"
    lbl_val = work_root / "labels" / "val"
    tform = transforms.Compose([transforms.ToTensor()])
    num_fg = len(classes)
    if model_type == "fasterrcnn":
        model = build_fasterrcnn(num_fg)
        ds_tr = YoloDetectionDataset(img_tr, lbl_tr, classes, transforms=tform, label_offset=CONFIG["frcnn"]["label_offset"])
        ds_val = YoloDetectionDataset(img_val, lbl_val, classes, transforms=tform, label_offset=CONFIG["frcnn"]["label_offset"])
        cat_offset = CONFIG["frcnn"]["cat_id_offset"]
        eval_conf = 0.0
    elif model_type == "retinanet":
        model = build_retinanet(num_fg)
        ds_tr = YoloDetectionDataset(img_tr, lbl_tr, classes, transforms=tform, label_offset=CONFIG["retinanet"]["label_offset"])
        ds_val = YoloDetectionDataset(img_val, lbl_val, classes, transforms=tform, label_offset=CONFIG["retinanet"]["label_offset"])
        cat_offset = CONFIG["retinanet"]["cat_id_offset"]
        eval_conf = CONFIG["retinanet"]["eval_conf"]
    else:
        raise ValueError("Unknown model_type")
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=CONFIG["tv"]["batch_train"], shuffle=True, num_workers=2, collate_fn=collate_fn)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=CONFIG["tv"]["batch_val"], shuffle=False, num_workers=2, collate_fn=collate_fn)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for imgs, tgts, _paths in tqdm(dl_tr, desc=f"{model_type} epoch {ep+1}/{epochs}"):
            imgs = [img.to(device) for img in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
            loss_dict = model(imgs, tgts)
            loss = sum(loss_dict.values())
            optim.zero_grad()
            loss.backward()
            optim.step()
            ep_loss += loss.item()
        sched.step()
        print(f"Epoch {ep+1}: loss={ep_loss / max(1, len(dl_tr)):.4f}")
    out_dir = work_root / f"torchvision_{model_type}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "model_final.pth"
    torch.save(model.state_dict(), ckpt)
    print("Saved", ckpt)
    # Eval
    dl_val_eval = torch.utils.data.DataLoader(ds_val, batch_size=CONFIG["tv"]["batch_val"], shuffle=False, num_workers=2, collate_fn=collate_fn)
    coco_gt, dets = evaluate_to_coco_json(
        model, dl_val_eval, device, work_root / "coco_val.json",
        score_thresh=eval_conf, category_id_offset=cat_offset, maxDets=CONFIG["coco"]["maxDets"]
    )
    metrics = coco_eval_from_results(coco_gt, dets, maxDets=CONFIG["coco"]["maxDets"])
    return model, metrics, out_dir

def visualize_model_preds_yolo(yolo_model, img_paths: List[Path], out_dir: Path, title_prefix: str, device: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    res = yolo_model.predict([str(p) for p in img_paths], imgsz=CONFIG["yolo"]["imgsz"], conf=0.25, device=0 if device.startswith("cuda") else "cpu", verbose=False)
    for p, r in zip(img_paths, res):
        vis = r.plot()[:, :, ::-1]
        out_path = out_dir / f"{title_prefix}_{p.stem}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

@torch.no_grad()
def visualize_model_preds_torchvision(model, img_paths: List[Path], out_dir: Path, title_prefix: str, device: str, class_names: List[str]):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    tform = transforms.Compose([transforms.ToTensor()])
    for p in img_paths:
        img = Image.open(p).convert("RGB")
        inp = tform(img).to(device)
        out = model([inp])[0]
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        boxes = out["boxes"].detach().cpu().numpy()
        scores = out["scores"].detach().cpu().numpy()
        labels = out["labels"].detach().cpu().numpy()
        for b, s, lb in zip(boxes, scores, labels):
            if s < 0.25:
                continue
            x0, y0, x1, y1 = map(int, b)
            # Map 0-based labels (RetinaNet) to 1-based for display name indexing
            cls_idx = lb if lb >= 1 else lb + 1
            cls_name = class_names[cls_idx - 1] if 1 <= cls_idx <= len(class_names) else str(lb)
            cv2.rectangle(img_cv, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(img_cv, f"{cls_name} {s:.2f}", (x0, max(0, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        out_path = out_dir / f"{title_prefix}_{p.stem}.png"
        cv2.imwrite(str(out_path), img_cv)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str, default=".", help="Project root directory")
    parser.add_argument("--dataset_dir", type=str, default="", help="Parent folder that contains images/ and annotations/")
    parser.add_argument("--skip_kaggle", action="store_true", help="Skip Kaggle download and use local dataset_dir or ./datasets")
    # Allow overriding CONFIG at CLI for quick experiments
    parser.add_argument("--epochs_yolo", type=int, default=CONFIG["yolo"]["epochs"])
    parser.add_argument("--epochs_tv", type=int, default=CONFIG["tv"]["epochs"])
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    outputs = root / CONFIG["outputs_dir"]
    outputs.mkdir(parents=True, exist_ok=True)
    device = device_info()

    # Resolve dataset location
    user_dir = args.dataset_dir or CONFIG["dataset_dir"]
    if user_dir:
        raw_ds = resolve_tomato_root(Path(user_dir).resolve())
        print("Using dataset_dir:", raw_ds)
    elif args.skip_kaggle:
        raw_ds = resolve_tomato_root(root / "datasets")
        print("Using local datasets (skip_kaggle):", raw_ds)
    else:
        try:
            raw_ds = download_variant10_dataset(root)
        except Exception as e:
            print("Kaggle download failed, using ./datasets; ensure it contains images/ and annotations/.", e)
            raw_ds = resolve_tomato_root(root / "datasets")

    # Validate
    sample_imgs = [p for p in raw_ds.rglob("*") if p.suffix.lower() in IMG_EXTS]
    print("Raw dataset root:", raw_ds)
    print("Found images:", len(sample_imgs))
    if len(sample_imgs) == 0:
        raise RuntimeError(f"No images found under {raw_ds}. Point --dataset_dir to a folder that contains images and annotations.")

    # Prepare data
    classes = CONFIG["classes"]
    work = root / CONFIG["work_dir"]
    data_yaml, work_root = prepare_yolo_and_coco(raw_ds, work, classes)

    # Sanity visuals
    sanity_dir = outputs / "sanity"
    sanity_dir.mkdir(parents=True, exist_ok=True)
    samples = list((work_root / "images" / "train").glob("*"))[:6]
    for p in samples:
        lbl = work_root / "labels" / "train" / f"{p.stem}.txt"
        img = draw_yolo_boxes(p, lbl, classes)
        cv2.imwrite(str(sanity_dir / f"train_{p.stem}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Train + eval YOLO
    CONFIG["yolo"]["epochs"] = args.epochs_yolo
    yolo_model, yolo_metrics = train_eval_yolo(data_yaml, work_root, device)

    # Train + eval Faster R-CNN
    CONFIG["tv"]["epochs"] = args.epochs_tv
    frcnn_model, frcnn_metrics, frcnn_dir = train_torchvision_detector(
        "fasterrcnn", work_root, device, classes,
        epochs=CONFIG["tv"]["epochs"], lr=CONFIG["tv"]["lr"], wd=CONFIG["tv"]["weight_decay"]
    )

    # Train + eval RetinaNet
    retina_model, retina_metrics, retina_dir = train_torchvision_detector(
        "retinanet", work_root, device, classes,
        epochs=CONFIG["tv"]["epochs"], lr=CONFIG["tv"]["lr"], wd=CONFIG["tv"]["weight_decay"]
    )

    # Consolidate metrics
    rows = [
        {"model": "YOLOv8n", "mAP_50_95": yolo_metrics["mAP_50_95"], "mAP_50": yolo_metrics["mAP_50"], "AR100": yolo_metrics["AR100"], "AP_tomato": yolo_metrics["AP_per_class"].get("tomato", float('nan'))},
        {"model": "Faster R-CNN", "mAP_50_95": frcnn_metrics["mAP_50_95"], "mAP_50": frcnn_metrics["mAP_50"], "AR100": frcnn_metrics["AR100"], "AP_tomato": frcnn_metrics["AP_per_class"].get("tomato", float('nan'))},
        {"model": "RetinaNet", "mAP_50_95": retina_metrics["mAP_50_95"], "mAP_50": retina_metrics["mAP_50"], "AR100": retina_metrics["AR100"], "AP_tomato": retina_metrics["AP_per_class"].get("tomato", float('nan'))},
    ]
    df = pd.DataFrame(rows)
    metrics_csv = outputs / "metrics_comparison.csv"
    df.to_csv(metrics_csv, index=False)
    print("Wrote", metrics_csv)

    # Visualizations
    vis_dir = outputs / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    val_imgs = list((work_root / "images" / "val").glob("*"))[:6]
    visualize_model_preds_yolo(yolo_model, val_imgs, vis_dir, "yolo", device)
    visualize_model_preds_torchvision(frcnn_model, val_imgs, vis_dir, "fasterrcnn", device, classes)
    visualize_model_preds_torchvision(retina_model, val_imgs, vis_dir, "retinanet", device, classes)
    print("Done.")

if __name__ == "__main__":
    main()
