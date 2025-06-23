import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

def get_args():
    parser = argparse.ArgumentParser(description="Grid search linear classifier on CIFAR-100 with pretrained ResNet")
    parser.add_argument('--arch', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                        help='ResNet backbone architecture')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr_list', type=float, nargs='+', required=True,
                        help='List of learning rates to try, e.g. --lr_list 0.1 0.01 0.001')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--data_dir', type=str, default='./data')
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Transforms ---
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # --- Data ---
    train_dataset = datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    best_overall_acc = 0.0
    best_lr = None

    # --- LR Grid Search ---
    for lr in tqdm(args.lr_list, desc="🔍 Grid Search over Learning Rates"):
        print(f"\nTrying learning rate: {lr:.5f}")

        # --- Model Setup ---
        model = getattr(models, args.arch)(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, 100)
        for param in model.fc.parameters():
            param.requires_grad = True
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9)

        best_acc = 0.0
        for epoch in trange(args.epochs, desc=f"LR={lr:.5f}"):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, targets in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            train_acc = 100. * correct / total
            print(f"Epoch {epoch+1}: Train Loss: {running_loss:.3f}, Train Accuracy: {train_acc:.2f}%")

            # --- Evaluation ---
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            test_acc = 100. * correct / total
            print(f"Epoch {epoch+1}: Test Accuracy: {test_acc:.2f}%")

            best_acc = max(best_acc, test_acc)

        print(f" Best Test Accuracy for LR={lr:.5f}: {best_acc:.2f}%")
        if best_acc > best_overall_acc:
            best_overall_acc = best_acc
            best_lr = lr

    print(f"\n Best LR: {best_lr:.5f} → Test Accuracy: {best_overall_acc:.2f}%")

if __name__ == '__main__':
    main()

