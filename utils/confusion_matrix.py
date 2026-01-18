import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.models import resnet18
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(model_path='trained_resnet18_cifar10.pth', batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transformations (same as used in training)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # Load CIFAR-10 test dataset
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Fix multiprocessing issue for Windows
    num_workers = 0 if torch.backends.mps.is_available() or torch.cuda.is_available() else 4
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Load the trained model safely
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)  # Adjust for 10 classes
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))  # Use weights_only=True
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # Collect predictions and true labels
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=testset.classes, yticklabels=testset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - ResNet18 on CIFAR-10')
    plt.show()

# Ensure safe execution on Windows
if __name__ == '__main__':
    plot_confusion_matrix()
