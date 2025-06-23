import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import time  # <-- Added for timing

# custom modules
from image_dataloader_from_file import image_loader

def accuracy(output, target, topk=(1, 5)):
    """Computes the top-k accuracy"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)  # top-k predictions
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # check against ground truth

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)  # count correct top-k
        res.append((correct_k / batch_size).item() * 100.0)  # return as percentage
    return res

def main(args):
    start_time = time.time()  # <-- Start timing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("In MAIN")

    # Load pre-trained model
    model = models.__dict__[args.arch](pretrained=True)
    model.eval()
    model.to(device)

    # Preprocessing matching training phase
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ImageNet val dataset
    #val_dataset = datasets.ImageFolder(root=args.data, transform=val_transforms)
    val_dataset = image_loader(args.data_file, args.data_root, input_transforms=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.workers, pin_memory=True)

    top1_total = 0.0
    top5_total = 0.0
    total = 0

    #with torch.no_grad():
    for images, targets in tqdm(val_loader, desc="Evaluating"):
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        top1, top5 = accuracy(outputs, targets, topk=(1, 5))

        batch_size = images.size(0)
        top1_total += top1 * batch_size
        top5_total += top5 * batch_size
        total += batch_size

    elapsed_time = time.time() - start_time  # <-- End timing

    print(f"Validation Top-1 Accuracy: {top1_total / total:.2f}%")
    print(f"Validation Top-5 Accuracy: {top5_total / total:.2f}%")
    print(f"Total evaluation time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ImageNet-1K Validation Accuracy")
    parser.add_argument("--data_root", type=str, required=True, help="Path to ImageNet val folder")
    parser.add_argument("--data_file", type=str, required=True, help="Path to ImageNet val annotation file.")
    parser.add_argument("--arch", type=str, default="resnet50", help="Model architecture (default: resnet50)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for DataLoader")
    parser.add_argument("--workers", type=int, default=4, help="Number of data loading workers")

    args = parser.parse_args()
    main(args)

