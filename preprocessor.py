import os
import numpy as np
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2#
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
import os

def load_and_preprocess(image_path, mask_path, target_size=224):
    # Load image and mask
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path)

    # Resize with padding (to maintain aspect ratio)
    def resize_with_padding(img, size):
        w, h = img.size
        scale = size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = img.resize((new_w, new_h))

        # Create padding
        pad_w, pad_h = size - new_w, size - new_h
        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2

        # Convert to numpy and add padding
        img_np = np.array(resized)
        padded_img = cv2.copyMakeBorder(img_np, pad_top, pad_bottom, pad_left, pad_right, 
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return Image.fromarray(padded_img)

    image = resize_with_padding(image, target_size)
    mask = resize_with_padding(mask, target_size)

    return image, mask

def extract_class_from_mask(mask):
    mask_np = np.array(mask)
    unique_labels, counts = np.unique(mask_np, return_counts=True)
    dominant_label = unique_labels[np.argmax(counts)]  # Most frequent label
    return dominant_label  # 0: Background, 1: Cat, 2: Dog

def augment_image_and_mask(image, mask):
    augmented = transform(image=np.array(image), mask=np.array(mask))
    return augmented["image"], augmented["mask"]

class CatDogDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_size=224):
        self.image_paths = sorted(os.listdir(image_dir))
        self.mask_paths = sorted(os.listdir(mask_dir))
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_paths[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_paths[idx])

        image, mask = load_and_preprocess(image_path, mask_path, self.target_size)
        label = extract_class_from_mask(mask)  # Get dominant class

        if self.transform:
            image, mask = augment_image_and_mask(image, mask)

        return image, torch.tensor(label, dtype=torch.long)
    
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard ImageNet normalization
    ToTensorV2(),
])

# Define dataset
dataset = CatDogDataset("C:/Users/louis/Documents/UNI4/CV/CW/Dataset/Dataset/TrainVal/color/", "C:/Users/louis/Documents/UNI4/CV/CW/Dataset/Dataset/TrainVal/color/", transform=transform)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
print(dataloader)

# Assign higher weight to less common classes
class_weights = torch.tensor([0.2, 0.4, 0.4])  # Adjust based on dataset distribution
loss_fn = CrossEntropyLoss(weight=class_weights)

# Define output folder for preprocessed dataset
output_dir = os.path.join(os.getcwd(), "processed_dataset")
os.makedirs(output_dir, exist_ok=True)

# Subfolders for images and labels
image_output_dir = os.path.join(output_dir, "color")
mask_output_dir = os.path.join(output_dir, "label")
os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(mask_output_dir, exist_ok=True)

print(f"Processed dataset will be saved in: {output_dir}")

def save_preprocessed_data(image_path, mask_path, output_image_dir, output_mask_dir, target_size=224):
    # Load and preprocess images/masks
    image, mask = load_and_preprocess(image_path, mask_path, target_size)

    # Get filename (without extension)
    filename = os.path.basename(image_path).split('.')[0]

    # Define output paths
    processed_image_path = os.path.join(output_image_dir, f"{filename}.jpg")
    processed_mask_path = os.path.join(output_mask_dir, f"{filename}.png")

    # Save images and masks
    image.save(processed_image_path, "JPEG")
    mask.save(processed_mask_path, "PNG")

    return processed_image_path, processed_mask_path

import csv

csv_path = os.path.join(output_dir, "labels.csv")

# Create a CSV file and write header
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "label"])  # Header

# Process all images in dataset and save results
image_dir = "C:/Users/louis/Documents/UNI4/CV/CW/Dataset/Dataset/TrainVal/color/"  # Input images directory
mask_dir = "C:/Users/louis/Documents/UNI4/CV/CW/Dataset/Dataset/TrainVal/label/"  # Input masks directory

# Open CSV file in append mode
with open(csv_path, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)

    for image_file in sorted(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, image_file.replace(".jpg", ".png"))  # Assume masks are .png
        
        if not os.path.exists(mask_path):
            #print(":o")
            continue  # Skip if mask is missing

        # Save preprocessed image and mask
        processed_image, processed_mask = save_preprocessed_data(image_path, mask_path, image_output_dir, mask_output_dir)

        # Extract classification label
        mask = Image.open(processed_mask)
        label = extract_class_from_mask(mask)

        # Save label in CSV file
        writer.writerow([image_file, label])

print(f"Processed images and labels saved in: {output_dir}")
print(f"Class labels saved in: {csv_path}")

