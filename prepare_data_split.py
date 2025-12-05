import os
import shutil
import glob
from pathlib import Path
import random

def main():
    data_root = Path("data")
    source_dir = data_root / "train"
    
    # Define output directories
    train_images_dir = data_root / "train" / "images"
    train_annotations_dir = data_root / "train" / "annotations"
    val_images_dir = data_root / "val" / "images"
    val_annotations_dir = data_root / "val" / "annotations"
    
    # Create directories
    for d in [train_images_dir, train_annotations_dir, val_images_dir, val_annotations_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # Get all image files
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(source_dir.glob(ext)))
        
    print(f"Found {len(image_files)} images.")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(image_files)
    
    split_idx = int(len(image_files) * 0.9)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Training set: {len(train_files)}")
    print(f"Validation set: {len(val_files)}")
    
    # Move files
    def move_files(file_list, dest_img_dir, dest_ann_dir):
        for img_path in file_list:
            # Determine JSON path
            json_path = source_dir / f"{img_path.stem}.json"
            
            if not json_path.exists():
                print(f"Warning: No annotation found for {img_path.name}, skipping.")
                continue
                
            # Move image
            shutil.move(str(img_path), str(dest_img_dir / img_path.name))
            
            # Move annotation
            shutil.move(str(json_path), str(dest_ann_dir / json_path.name))

    print("Moving training files...")
    move_files(train_files, train_images_dir, train_annotations_dir)
    
    print("Moving validation files...")
    move_files(val_files, val_images_dir, val_annotations_dir)
    
    print("Data preparation complete.")

if __name__ == "__main__":
    main()
