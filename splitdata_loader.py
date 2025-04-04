import json

# Reload dataset
dataset = SegmentationDataset(image_dir, mask_dir, csv_path)

# Load indices from files
with open("train_indices.json", "r") as f:
    train_indices = json.load(f)

with open("val_indices.json", "r") as f:
    val_indices = json.load(f)

# Recreate the subsets
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)
