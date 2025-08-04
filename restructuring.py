import os
import shutil
import glob

source_dir = '.' 
dest_dir = 'brain_tumor_dataset_yolo'

class_map = {
    0: 0,  # Glioma 
    1: 1,  # Meningioma
    3: 2   # Pituitary (gantiin 'No Tumor')
}

class_folders_to_process = ['Glioma', 'Meningioma', 'Pituitary']

# 1. Create the destination directory structure.
for split in ['train', 'val']:
    os.makedirs(os.path.join(dest_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'labels', split), exist_ok=True)

def process_split(split_name):
    print(f"\n- Processing {split_name.upper()}")
    source_split_dir = os.path.join(source_dir, split_name.capitalize())

    # 2. Process folders with actual tumors
    for class_name in class_folders_to_process:
        class_dir = os.path.join(source_split_dir, class_name)
        
        # Copy all images for the class
        for img_path in glob.glob(os.path.join(class_dir, 'images', '*.jpg')):
            shutil.copy(img_path, os.path.join(dest_dir, 'images', split_name))

        # Find, remap, and copy all labels for the class
        for label_path in glob.glob(os.path.join(class_dir, 'labels', '*.txt')):
            new_lines = []
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    original_class_id = int(parts[0])
                    if original_class_id in class_map:
                        new_class_id = class_map[original_class_id]
                        new_line = f"{new_class_id} {' '.join(parts[1:])}"
                        new_lines.append(new_line)
            
            if new_lines:
                # Get the base filename to save it in the destination
                base_filename = os.path.basename(label_path)
                with open(os.path.join(dest_dir, 'labels', split_name, base_filename), 'w') as f:
                    f.write('\n'.join(new_lines))

    # 3. Process "No Tumor" folder for background images
    print("Processing 'No Tumor' for background images (labels are ignored)...")
    no_tumor_img_dir = os.path.join(source_split_dir, 'No Tumor', 'images')
    for img_path in glob.glob(os.path.join(no_tumor_img_dir, '*.jpg')):
        shutil.copy(img_path, os.path.join(dest_dir, 'images', split_name))

    print(f"Finished processing '{split_name.upper()}' split.")

process_split('train')
process_split('val')