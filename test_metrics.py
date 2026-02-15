import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import segmentation_models_pytorch as smp
from tqdm import tqdm
import matplotlib.pyplot as plt

plt.switch_backend('Agg')

# ============================================================================
# Configuration
# ============================================================================

class Config:
    BATCH_SIZE = 8
    IMAGE_HEIGHT = 224
    IMAGE_WIDTH = 384
    ENCODER = 'mobilenet_v2'
    NUM_CLASSES = 10
    DEVICE = torch.device('cpu')


# ============================================================================
# Mask Conversion
# ============================================================================

value_map = {
    0: 0,        # background
    100: 1,      # Trees
    200: 2,      # Lush Bushes
    300: 3,      # Dry Grass
    500: 4,      # Dry Bushes
    550: 5,      # Ground Clutter
    600: 6,      # Flowers
    700: 7,      # Logs
    800: 8,      # Rocks
    7100: 9,     # Landscape
    10000: 9     # Sky (same as landscape)
}

class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Flowers', 'Logs', 'Rocks', 'Landscape/Sky'
]

color_palette = np.array([
    [0, 0, 0],        # Background - black
    [34, 139, 34],    # Trees - forest green
    [0, 255, 0],      # Lush Bushes - lime
    [210, 180, 140],  # Dry Grass - tan
    [139, 90, 43],    # Dry Bushes - brown
    [128, 128, 0],    # Ground Clutter - olive
    [255, 0, 255],    # Flowers - magenta
    [139, 69, 19],    # Logs - saddle brown
    [128, 128, 128],  # Rocks - gray
    [135, 206, 235],  # Landscape/Sky - sky blue
], dtype=np.uint8)


def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


def mask_to_color(mask):
    """Convert class mask to colored RGB image."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(Config.NUM_CLASSES):
        color_mask[mask == class_id] = color_palette[class_id]
    return color_mask


# ============================================================================
# Test Dataset (With Ground Truth)
# ============================================================================

class TestDatasetWithGT(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.mask_transform = mask_transform
        self.data_ids = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = convert_mask(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.mask_transform(mask) * 255

        return image, mask, data_id


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=10):
    """Compute IoU for each class and return mean IoU + per-class IoU."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    iou_per_class = []
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).cpu().numpy())

    return np.nanmean(iou_per_class), iou_per_class


def compute_pixel_accuracy(pred, target):
    """Compute pixel accuracy."""
    pred_classes = torch.argmax(pred, dim=1)
    return (pred_classes == target).float().mean().cpu().numpy()


# ============================================================================
# Visualization
# ============================================================================

def save_comparison_samples(images, gt_masks, pred_masks, filenames, output_dir, num_samples=10):
    """Save sample comparisons."""
    os.makedirs(output_dir, exist_ok=True)
    
    num_samples = min(num_samples, len(images))
    
    for i in range(num_samples):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        axes[0].imshow(img)
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')
        
        # Ground Truth
        gt_colored = mask_to_color(gt_masks[i].cpu().numpy().astype(np.uint8))
        axes[1].imshow(gt_colored)
        axes[1].set_title('Ground Truth', fontsize=14)
        axes[1].axis('off')
        
        # Prediction
        pred_colored = mask_to_color(pred_masks[i].cpu().numpy().astype(np.uint8))
        axes[2].imshow(pred_colored)
        axes[2].set_title('Prediction', fontsize=14)
        axes[2].axis('off')
        
        plt.suptitle(f'Sample: {filenames[i]}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_{i+1:03d}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Saved {num_samples} comparison visualizations")


def save_per_class_iou_chart(class_iou, output_path):
    """Save bar chart of per-class IoU."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    valid_iou = [iou if not np.isnan(iou) else 0 for iou in class_iou]
    colors_normalized = [color_palette[i] / 255 for i in range(Config.NUM_CLASSES)]
    
    bars = ax.bar(range(Config.NUM_CLASSES), valid_iou, color=colors_normalized, 
                   edgecolor='black', linewidth=1.5)
    
    ax.set_xticks(range(Config.NUM_CLASSES))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel('IoU Score', fontsize=12)
    ax.set_title(f'Per-Class IoU (Mean: {np.nanmean(class_iou):.4f})', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.axhline(y=np.nanmean(class_iou), color='red', linestyle='--', linewidth=2, label='Mean IoU')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, iou) in enumerate(zip(bars, valid_iou)):
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{iou:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved per-class IoU chart to '{output_path}'")


# ============================================================================
# Main Test Function
# ============================================================================

def main():
    print("=" * 80)
    print("TESTING WITH GROUND TRUTH - IoU CALCULATION")
    print("=" * 80)
    print(f"Device: {Config.DEVICE}")
    print(f"Model: DeepLabV3+ with {Config.ENCODER}")
    print(f"Image Size: {Config.IMAGE_HEIGHT}x{Config.IMAGE_WIDTH}")
    print("=" * 80 + "\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load model
    print("Loading trained model...")
    model = smp.DeepLabV3Plus(
        encoder_name=Config.ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=Config.NUM_CLASSES,
    )
    
    model_path = os.path.join(script_dir, 'best_segmentation_model.pth')
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at '{model_path}'")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    model.eval()
    print(f"✓ Model loaded successfully\n")

    # Transforms
    test_transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH), 
                         interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])

    # Load test dataset with ground truth
    test_data_dir = os.path.join(script_dir, 'test_public_80')
    
    if not os.path.exists(test_data_dir):
        print(f"ERROR: Test folder not found at '{test_data_dir}'")
        return
    
    print(f"Loading test dataset from: {test_data_dir}")
    test_dataset = TestDatasetWithGT(
        data_dir=test_data_dir,
        transform=test_transform,
        mask_transform=mask_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                            shuffle=False, num_workers=0, drop_last=False)
    
    print(f"✓ Loaded {len(test_dataset)} test images with ground truth\n")

    # Create output directory
    output_dir = os.path.join(script_dir, 'test_evaluation')
    comparisons_dir = os.path.join(output_dir, 'comparisons')
    os.makedirs(comparisons_dir, exist_ok=True)

    # Run evaluation
    print("Running evaluation on test set...")
    all_iou_scores = []
    all_pixel_acc = []
    all_class_iou = []
    
    sample_images = []
    sample_gt = []
    sample_pred = []
    sample_names = []
    
    with torch.no_grad():
        for imgs, labels, filenames in tqdm(test_loader, desc="Testing", unit="batch"):
            imgs, labels = imgs.to(Config.DEVICE), labels.to(Config.DEVICE)
            labels = labels.squeeze(dim=1).long()

            # Forward pass
            outputs = model(imgs)
            predictions = torch.argmax(outputs, dim=1)

            # Calculate metrics
            iou, class_iou = compute_iou(outputs, labels, num_classes=Config.NUM_CLASSES)
            pixel_acc = compute_pixel_accuracy(outputs, labels)

            all_iou_scores.append(iou)
            all_pixel_acc.append(pixel_acc)
            all_class_iou.append(class_iou)
            
            # Collect samples for visualization
            if len(sample_images) < 10:
                for img, gt, pred, fname in zip(imgs, labels, predictions, filenames):
                    if len(sample_images) < 10:
                        sample_images.append(img.cpu())
                        sample_gt.append(gt.cpu())
                        sample_pred.append(pred.cpu())
                        sample_names.append(fname)

    # Calculate final metrics
    mean_iou = np.nanmean(all_iou_scores)
    mean_pixel_acc = np.mean(all_pixel_acc)
    mean_class_iou = np.nanmean(all_class_iou, axis=0)

    # Print results
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 80)
    print(f"Mean IoU:           {mean_iou:.4f} ({mean_iou*100:.2f}%)")
    print(f"Pixel Accuracy:     {mean_pixel_acc:.4f} ({mean_pixel_acc*100:.2f}%)")
    print("=" * 80)
    print("\nPer-Class IoU:")
    print("-" * 80)
    for i, (name, iou) in enumerate(zip(class_names, mean_class_iou)):
        iou_str = f"{iou:.4f} ({iou*100:.2f}%)" if not np.isnan(iou) else "N/A"
        print(f"  {name:<20}: {iou_str}")
    print("=" * 80)

    # Save results to file
    results_file = os.path.join(output_dir, 'test_results.txt')
    with open(results_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TEST SET EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: DeepLabV3+ with {Config.ENCODER}\n")
        f.write(f"Test Images: {len(test_dataset)}\n")
        f.write(f"Image Size: {Config.IMAGE_HEIGHT}x{Config.IMAGE_WIDTH}\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean IoU:           {mean_iou:.4f} ({mean_iou*100:.2f}%)\n")
        f.write(f"Pixel Accuracy:     {mean_pixel_acc:.4f} ({mean_pixel_acc*100:.2f}%)\n\n")
        
        f.write("PER-CLASS IoU:\n")
        f.write("-" * 80 + "\n")
        for i, (name, iou) in enumerate(zip(class_names, mean_class_iou)):
            iou_str = f"{iou:.4f} ({iou*100:.2f}%)" if not np.isnan(iou) else "N/A"
            f.write(f"  {name:<20}: {iou_str}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("COMPARISON WITH VALIDATION:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Validation IoU:     0.4149 (from training)\n")
        f.write(f"Test IoU:           {mean_iou:.4f}\n")
        diff = mean_iou - 0.4149
        f.write(f"Difference:         {diff:+.4f} ({diff*100:+.2f}%)\n")
    
    print(f"\n✓ Saved detailed results to '{results_file}'")

    # Create visualizations
    print("\nCreating visualizations...")
    save_comparison_samples(sample_images, sample_gt, sample_pred, sample_names, 
                           comparisons_dir, num_samples=10)
    
    chart_path = os.path.join(output_dir, 'per_class_iou.png')
    save_per_class_iou_chart(mean_class_iou, chart_path)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"\n📊 TEST IoU SCORE: {mean_iou:.4f} ({mean_iou*100:.2f}%)")
    print(f"📁 Results saved in: {output_dir}/")
    print(f"  - test_results.txt       : Detailed metrics")
    print(f"  - per_class_iou.png      : IoU bar chart")
    print(f"  - comparisons/           : Sample predictions vs ground truth")
    print("=" * 80)


if __name__ == "__main__":
    main()