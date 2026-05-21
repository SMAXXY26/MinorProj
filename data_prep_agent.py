import os
import cv2
import yaml
import shutil
import urllib.request
import zipfile
import imagehash
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict

def init_directories(base_dir="data"):
    print("1. Initializing directories...")
    base = Path(base_dir)
    dirs = [
        base / "images" / "train",
        base / "images" / "val",
        base / "images" / "test",
        base / "labels" / "train",
        base / "labels" / "val",
        base / "labels" / "test",
        base / "classifier_crops" / "knife",
        base / "classifier_crops" / "pistol",
        base / "classifier_crops" / "rifle",
        base / "negatives" / "images",
        base / "raw" / "processed"
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return base

def download_data(base):
    print("2. Downloading data...")
    raw_dir = base / "raw"
    
    # FiftyOne
    try:
        import fiftyone.zoo as foz
        import fiftyone as fo
        print("Downloading FiftyOne open-images-v7...")
        # Download separately to control exact counts
        classes_counts = [("Knife", 3000), ("Handgun", 4000), ("Rifle", 1500)]
        for cls_name, count in classes_counts:
            dataset = foz.load_zoo_dataset(
                "open-images-v7",
                split="train",
                label_types=["detections"],
                classes=[cls_name],
                max_samples=count,
                dataset_dir=str(raw_dir / f"fiftyone_{cls_name.lower()}")
            )
            dataset.export(
                export_dir=str(raw_dir / f"fiftyone_export_{cls_name.lower()}"),
                dataset_type=fo.types.YOLOv5Dataset,
                classes=[cls_name]
            )
    except Exception as e:
        print(f"[WARN] FiftyOne download failed: {e}")

    # Roboflow
    print("Downloading Roboflow datasets...")
    rf_num = os.getenv("ROBOFLOW_API_KEY")
    if rf_num and rf_num != "your_api_key_here":
        from roboflow import Roboflow
        rf = Roboflow(api_key=rf_num)
        projects = ["weapons-detection", "knife-detection-9bbzc", "gun-object-detection", "pistol-detection-hpvol"]
        for p in projects:
            try:
                # We assume the API key is linked to a default workspace or the project is public.
                # Since workspace is required by the python SDK usually, we can infer from the project name optionally,
                # but rf.workspace().project(p) works if it's in the default workspace.
                # Let's try downloading via URL request or generic API if workspace is missing
                proj = rf.workspace().project(p)
                proj.version(1).download("yolov8", location=str(raw_dir / f"rf_{p}"), overwrite=True)
            except Exception as e:
                print(f"[WARN] Roboflow download for {p} failed: {e}")
    else:
        print("[WARN] No ROBOFLOW_API_KEY found, skipping Roboflow.")

    # COCO
    print("Downloading COCO (knife category)...")
    try:
        from pycocotools.coco import COCO
        coco_dir = raw_dir / "coco"
        coco_dir.mkdir(exist_ok=True)
        anno_zip = coco_dir / "annotations.zip"
        if not anno_zip.exists():
            print("Downloading COCO annotations...")
            urllib.request.urlretrieve("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", anno_zip)
            with zipfile.ZipFile(anno_zip, 'r') as zip_ref:
                zip_ref.extractall(coco_dir)
        
        coco = COCO(str(coco_dir / "annotations" / "instances_train2017.json"))
        catIds = coco.getCatIds(catNms=['knife'])
        if catIds:
            imgIds = coco.getImgIds(catIds=catIds)
            images = coco.loadImgs(imgIds)
            coco_img_dir = coco_dir / "images"
            coco_lbl_dir = coco_dir / "labels"
            coco_img_dir.mkdir(exist_ok=True)
            coco_lbl_dir.mkdir(exist_ok=True)
            
            for img in tqdm(images, desc="COCO Images"):
                url = img['coco_url']
                img_path = coco_img_dir / img['file_name']
                if not img_path.exists():
                    urllib.request.urlretrieve(url, img_path)
                
                # Create label file
                annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
                anns = coco.loadAnns(annIds)
                with open(coco_lbl_dir / (img_path.stem + ".txt"), "w") as f:
                    for ann in anns:
                        # COCO bbox is [x_min, y_min, width, height]
                        # YOLO is [x_center, y_center, width, height] normalized
                        x, y, w, h = ann['bbox']
                        x_center = (x + w/2) / img['width']
                        y_center = (y + h/2) / img['height']
                        w_norm = w / img['width']
                        h_norm = h / img['height']
                        # 0 is our class for knife
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
    except Exception as e:
        print(f"[WARN] COCO download failed: {e}")

def remap_and_filter(base):
    print("3. Remapping & Filtering Logic...")
    raw_dir = base / "raw"
    processed_dir = base / "raw" / "processed"
    
    class_map = {
        "knife": 0, "knives": 0, "blade": 0, "kitchen_knife": 0,
        "gun": 1, "handgun": 1, "pistol": 1, "firearm": 1, "revolver": 1,
        "rifle": 2, "sniper": 2, "shotgun": 2, "assault_rifle": 2
    }
    
    # Process all datasets in raw_dir
    img_counter = 0
    for root, dirs, files in os.walk(raw_dir):
        if "processed" in root: continue
        
        # Check if this folder has a data.yaml
        yaml_path = Path(root) / "data.yaml"
        src_names = {}
        if yaml_path.exists():
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
                if "names" in data:
                    if isinstance(data["names"], list):
                        src_names = {i: n.lower() for i, n in enumerate(data["names"])}
                    elif isinstance(data["names"], dict):
                        src_names = {i: n.lower() for i, n in data["names"].items()}
        elif "fiftyone" in str(root):
            src_names = {0: "knife", 1: "handgun", 2: "rifle"}
        
        # If no yaml and not COCO, try to infer from parent if it's images/labels
        if "images" in root:
            # Match with labels
            lbl_dir = Path(root).parent / "labels"
            if not lbl_dir.exists():
                lbl_dir = Path(root).parent.parent / "labels" / Path(root).name # train/val splits
            if not lbl_dir.exists():
                lbl_dir = Path(root).with_name("labels")
                
            for img_file in files:
                if not img_file.lower().endswith((".jpg", ".jpeg", ".png")): continue
                
                img_path = Path(root) / img_file
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                
                if not lbl_path.exists(): continue
                
                # If COCO, it's already mapped to 0
                if "coco" in str(root).lower():
                    local_src_names = {0: "knife"}
                else:
                    local_src_names = src_names
                
                try:
                    img = Image.open(img_path)
                    img_w, img_h = img.size
                except:
                    continue
                
                valid_boxes = []
                with open(lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5: continue
                        cls_id = int(parts[0])
                        
                        src_name = local_src_names.get(cls_id, "").lower()
                        our_cls = class_map.get(src_name)
                        
                        if our_cls is None:
                            # Try substring matching
                            for k, v in class_map.items():
                                if k in src_name:
                                    our_cls = v
                                    break
                                    
                        if our_cls is None and "coco" in str(root).lower():
                            our_cls = 0
                            
                        if our_cls is not None:
                            cx, cy, bw, bh = map(float, parts[1:5])
                            px_w = bw * img_w
                            px_h = bh * img_h
                            area_ratio = bw * bh
                            
                            # Filter: <10px or >95% area
                            if px_w >= 10 and px_h >= 10 and area_ratio <= 0.95:
                                valid_boxes.append(f"{our_cls} {cx} {cy} {bw} {bh}")
                
                if valid_boxes:
                    dst_img = processed_dir / f"img_{img_counter:06d}.jpg"
                    dst_lbl = processed_dir / f"img_{img_counter:06d}.txt"
                    shutil.copy2(img_path, dst_img)
                    with open(dst_lbl, "w") as f:
                        f.write("\n".join(valid_boxes) + "\n")
                    img_counter += 1

def deduplicate_and_crop(base):
    print("4. Deduplication & Segmentation...")
    processed_dir = base / "raw" / "processed"
    
    images = list(processed_dir.glob("*.jpg"))
    hashes = {}
    
    for img_path in tqdm(images, desc="Hashing"):
        try:
            hashes[img_path] = imagehash.phash(Image.open(img_path))
        except:
            pass
            
    paths = list(hashes.keys())
    to_remove = set()
    
    print("Finding duplicates...")
    for i in tqdm(range(len(paths)), desc="Comparing hashes"):
        if paths[i] in to_remove: continue
        h1 = hashes[paths[i]]
        for j in range(i+1, len(paths)):
            if paths[j] in to_remove: continue
            if h1 - hashes[paths[j]] < 8:
                sz1 = Image.open(paths[i]).size
                sz2 = Image.open(paths[j]).size
                if sz1[0]*sz1[1] >= sz2[0]*sz2[1]:
                    to_remove.add(paths[j])
                else:
                    to_remove.add(paths[i])
                    break
                    
    for p in to_remove:
        lbl = p.with_suffix(".txt")
        p.unlink()
        if lbl.exists(): lbl.unlink()
        
    print(f"Removed {len(to_remove)} duplicates. Proceeding to crop...")

def split_and_config(base):
    print("5. Splitting Dataset...")
    processed_dir = base / "raw" / "processed"
    images = sorted(list(processed_dir.glob("*.jpg")))
    
    # Stratified split requires knowing the classes in each image
    # Simplification: we just split randomly 75/15/10. For full stratification, we would multi-label stratify.
    # We will stratify based on the most frequent class in the image.
    class_labels = []
    valid_images = []
    
    for img_path in images:
        lbl_path = img_path.with_suffix(".txt")
        if lbl_path.exists():
            with open(lbl_path) as f:
                lines = f.readlines()
                if lines:
                    c = int(lines[0].split()[0]) # take first class
                    class_labels.append(c)
                    valid_images.append(img_path)
                    
    if not valid_images:
        print("[WARN] No valid images to split.")
        return
        
    train_x, temp_x, train_y, temp_y = train_test_split(valid_images, class_labels, test_size=0.25, stratify=class_labels, random_state=42)
    val_x, test_x, val_y, test_y = train_test_split(temp_x, temp_y, test_size=0.40, stratify=temp_y, random_state=42) # 0.25 * 0.6 = 0.15 val, 0.25 * 0.4 = 0.1 test
    
    splits = {"train": train_x, "val": val_x, "test": test_x}
    
    for split_name, imgs in splits.items():
        img_dir = base / "images" / split_name
        lbl_dir = base / "labels" / split_name
        
        for img_path in tqdm(imgs, desc=f"Copying {split_name}"):
            shutil.copy2(img_path, img_dir / img_path.name)
            lbl_path = img_path.with_suffix(".txt")
            shutil.copy2(lbl_path, lbl_dir / lbl_path.name)
            
            # Crop train set
            if split_name == "train":
                img = cv2.imread(str(img_path))
                if img is None: continue
                img_h, img_w = img.shape[:2]
                
                with open(lbl_path) as f:
                    for idx, line in enumerate(f):
                        parts = line.strip().split()
                        c = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        
                        x_center, y_center = cx * img_w, cy * img_h
                        width, height = bw * img_w, bh * img_h
                        
                        # 15% padding
                        width *= 1.15
                        height *= 1.15
                        
                        x1 = max(0, int(x_center - width/2))
                        y1 = max(0, int(y_center - height/2))
                        x2 = min(img_w, int(x_center + width/2))
                        y2 = min(img_h, int(y_center + height/2))
                        
                        crop = img[y1:y2, x1:x2]
                        if crop.size > 0:
                            crop_resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
                            c_name = {0: "knife", 1: "pistol", 2: "rifle"}[c]
                            crop_path = base / "classifier_crops" / c_name / f"{img_path.stem}_{idx}.jpg"
                            cv2.imwrite(str(crop_path), crop_resized)

    # Config
    yaml_content = """path: data
train: images/train
val: images/val
test: images/test
nc: 3
names: {0: knife, 1: pistol, 2: rifle}"""

    with open(base / "dataset.yaml", "w") as f:
        f.write(yaml_content)

def validate_dataset(base):
    print("6. Final Validation...")
    splits = ["train", "val", "test"]
    summary = defaultdict(lambda: defaultdict(int))
    
    for split in splits:
        img_dir = base / "images" / split
        lbl_dir = base / "labels" / split
        
        imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        lbls = list(lbl_dir.glob("*.txt"))
        
        assert len(imgs) == len(lbls), f"{split}: {len(imgs)} images vs {len(lbls)} labels"
        
        for lbl_path in lbls:
            with open(lbl_path) as f:
                for line in f:
                    c = int(line.split()[0])
                    assert c <= 2, f"Invalid class ID {c} in {lbl_path}"
                    summary[split][c] += 1
                    
    print("\nFinal Validation Summary:")
    print(f"{'Split':<10} | {'Knife (0)':<10} | {'Pistol (1)':<10} | {'Rifle (2)':<10}")
    print("-" * 50)
    for split in splits:
        print(f"{split:<10} | {summary[split][0]:<10} | {summary[split][1]:<10} | {summary[split][2]:<10}")

if __name__ == "__main__":
    base = init_directories("data")
    download_data(base)
    remap_and_filter(base)
    deduplicate_and_crop(base)
    split_and_config(base)
    validate_dataset(base)
    print("Data preparation complete!")
