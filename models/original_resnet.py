import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

def train_original_resnet(num_epochs=10, batch_size=64, learning_rate=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fix: Set CUDA device correctly
    if torch.cuda.is_available():
        torch.cuda.set_device(torch.cuda.current_device())  #  Get actual device index
        print(f" Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")


    torch.backends.cudnn.benchmark = True  #  Enable fast GPU computation
    #  Data Preparation (CIFAR-10)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    #  Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    #  Ensure Model is Initialized BEFORE Printing
    try:
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 10)
        model.to(device)  # Move to GPU
        print(f" Model is on: {next(model.parameters()).device}")  # Debug GPU usage
    except Exception as e:
        print(f" Model initialization failed: {e}")
        return

    #  Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scaler = torch.amp.GradScaler('cuda')  #  Enable Mixed Precision

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):  #  Use Mixed Precision
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)
        print(f" Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100.0 * correct / total
        print(f" Test Accuracy after epoch {epoch + 1}: {acc:.2f}%")

    torch.save(model.state_dict(), 'trained_resnet18_cifar10.pth')
    print(" Model saved: 'trained_resnet18_cifar10.pth'")
    return model

if __name__ == '__main__':
    train_original_resnet()
