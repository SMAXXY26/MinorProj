import os
from pathlib import Path
from PIL import Image

def crop_objects(base_out_dir):
    # Resolve the project root dynamically assuming the script is in MinorProj/newtest/congestion
    project_root = Path(__file__).resolve().parent.parent.parent
    
    out_dir = project_root / base_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for cls_name in ['Knife', 'Pistol', 'Rifle']:
        (out_dir / cls_name).mkdir(parents=True, exist_ok=True)

    cnt = 0
    # dataset 1: contains Grenade, Knife, Missile, Pistol, Rifle
    ds1_path = project_root / 'weapon_detection/weapon detection.v1i.yolov8'
    if ds1_path.exists():
        print(f"Processing {ds1_path}...")
        classes_map = {1: 'Knife', 3: 'Pistol', 4: 'Rifle'}
        for split in ['train', 'valid', 'test']:
            lbl_dir = ds1_path / split / 'labels'
            img_dir = ds1_path / split / 'images'
            if not lbl_dir.exists(): continue
            for lbl_file in lbl_dir.glob('*.txt'):
                img_file = img_dir / (lbl_file.stem + '.jpg')
                if not img_file.exists(): continue
                
                try:
                    img = Image.open(str(img_file))
                except Exception:
                    continue
                
                w, h = img.size
                
                with open(lbl_file, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) < 5: continue
                        cls_id = int(parts[0])
                        if cls_id in classes_map:
                            x_center = float(parts[1]) * w
                            y_center = float(parts[2]) * h
                            width = float(parts[3]) * w
                            height = float(parts[4]) * h
                            
                            x1 = max(0, int(x_center - width/2))
                            y1 = max(0, int(y_center - height/2))
                            x2 = min(w, int(x_center + width/2))
                            y2 = min(h, int(y_center + height/2))
                            
                            if x2 > x1 and y2 > y1:
                                crop_img = img.crop((x1, y1, x2, y2))
                                if crop_img.mode in ("RGBA", "P"): 
                                    crop_img = crop_img.convert("RGB")
                                out_path = out_dir / classes_map[cls_id] / f"ds1_{cnt}.jpg"
                                crop_img.save(str(out_path))
                                cnt += 1

    # dataset 2: Pistols only
    ds2_path = project_root / 'weapon_detection/Pistols.v1-resize-416x416.yolov8'
    if ds2_path.exists():
        print(f"Processing {ds2_path}...")
        for split in ['train', 'valid', 'test']:
            lbl_dir = ds2_path / split / 'labels'
            img_dir = ds2_path / split / 'images'
            if not lbl_dir.exists(): continue
            for lbl_file in lbl_dir.glob('*.txt'):
                img_file = img_dir / (lbl_file.stem + '.jpg')
                if not img_file.exists(): continue
                
                try:
                    img = Image.open(str(img_file))
                except Exception:
                    continue
                w, h = img.size
                
                with open(lbl_file, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) < 5: continue
                        cls_id = int(parts[0])
                        if cls_id == 0:
                            x_center = float(parts[1]) * w
                            y_center = float(parts[2]) * h
                            width = float(parts[3]) * w
                            height = float(parts[4]) * h
                            
                            x1 = max(0, int(x_center - width/2))
                            y1 = max(0, int(y_center - height/2))
                            x2 = min(w, int(x_center + width/2))
                            y2 = min(h, int(y_center + height/2))
                            
                            if x2 > x1 and y2 > y1:
                                crop_img = img.crop((x1, y1, x2, y2))
                                if crop_img.mode in ("RGBA", "P"): 
                                    crop_img = crop_img.convert("RGB")
                                out_path = out_dir / 'Pistol' / f"ds2_{cnt}.jpg"
                                crop_img.save(str(out_path))
                                cnt += 1
                                
    print(f"Total cropped images saved to {out_dir}: {cnt}")

if __name__ == '__main__':
    crop_objects('newtest/cropped_classes')
