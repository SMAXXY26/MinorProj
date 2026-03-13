import os
import shutil
import random
from pathlib import Path

def split_dataset(src_dir, dest_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)
    
    classes = [d.name for d in src_path.iterdir() if d.is_dir()]
    print(f"Found classes: {classes}")
    
    splits = ['train', 'val', 'test']
    for split in splits:
        for cls_name in classes:
            (dest_path / split / cls_name).mkdir(parents=True, exist_ok=True)
            
    for cls_name in classes:
        cls_dir = src_path / cls_name
        images = [f for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in ('.jpg', '.png', '.jpeg')]
        random.shuffle(images)
        
        num_images = len(images)
        train_end = int(train_ratio * num_images)
        val_end = train_end + int(val_ratio * num_images)
        
        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]
        
        def copy_imgs(img_list, split_name):
            for img in img_list:
                shutil.copy2(img, dest_path / split_name / cls_name / img.name)
        
        copy_imgs(train_imgs, 'train')
        copy_imgs(val_imgs, 'val')
        copy_imgs(test_imgs, 'test')
        
        print(f"Class '{cls_name}': {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")

if __name__ == '__main__':
    random.seed(42)
    # the cropped_classes are already gathered
    split_dataset(
        src_dir='cropped_classes', 
        dest_dir='dataset_split', 
        train_ratio=0.7, 
        val_ratio=0.15, 
        test_ratio=0.15
    )
    print("Dataset setup for testing complete!")
