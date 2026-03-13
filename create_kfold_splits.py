import os
import glob
from pathlib import Path
import random
import yaml

def create_kfold_splits(base_dir, k=5):
    base_path = Path(base_dir).resolve()
    print(f"Base data path: {base_path}")
    
    # Gathering all images from the different datasets and splits (train/valid/test)
    image_paths = []
    
    # Find all images by globbing through weapon_detection directories, look inside */*/*.jpg
    search_pattern = str(base_path / "**" / "*.jpg")
    for img_path in glob.iglob(search_pattern, recursive=True):
        image_paths.append(img_path)
        
    print(f"Total images found: {len(image_paths)}")
    
    if len(image_paths) == 0:
        print("No images found. Exiting.")
        return

    # Seed removed for fully random shuffling
    random.shuffle(image_paths)
    
    n = len(image_paths)
    fold_size = n // k
    splits = []
    
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i < k - 1 else n
        val_idx = list(range(start, end))
        train_idx = list(range(0, start)) + list(range(end, n))
        splits.append((train_idx, val_idx))
    
    kfold_dir = base_path / f"{k}fold_cv"
    kfold_dir.mkdir(parents=True, exist_ok=True)
    
    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        fold_dir = kfold_dir / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)
        
        train_file = fold_dir / "train.txt"
        val_file = fold_dir / "val.txt"
        
        with open(train_file, 'w') as f_train:
            for idx in train_idx:
                f_train.write(f"{image_paths[idx]}\n")
                
        with open(val_file, 'w') as f_val:
            for idx in val_idx:
                f_val.write(f"{image_paths[idx]}\n")
        
        # Create dataset yaml for this fold
        yaml_content = {
            'path': str(fold_dir),
            'train': str(train_file),
            'val': str(val_file),
            'nc': 1,
            'names': ['weapon']
        }
        
        yaml_file = fold_dir / "data.yaml"
        with open(yaml_file, 'w') as f_yaml:
            yaml.dump(yaml_content, f_yaml, sort_keys=False)
            
        print(f"Fold {fold} prepared: {len(train_idx)} train, {len(val_idx)} val -> {yaml_file}")

if __name__ == "__main__":
    create_kfold_splits("./weapon_detection", k=5)
