import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

DEVICE = get_device()
print(f"Using device: {DEVICE}")

class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(2),                             
            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2),                              
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class CifarCNN(nn.Module):
    def __init__(self):
        super(CifarCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_mnist_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) 
    ])
    train_set = torchvision.datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
    test_set  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader

def get_cifar_loaders(batch_size=128):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),           
        transforms.RandomCrop(32, padding=4),        
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)) 
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=train_transform)
    test_set  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), 100. * correct / total

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), 100. * correct / total

def train_model(model, train_loader, test_loader, epochs, lr=0.001, name="Model", optimizer_type='adam'):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    if optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print(f"\n{'='*50}")
    print(f"  {name} 학습 시작 (optimizer={optimizer_type})")
    print(f"{'='*50}")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        scheduler.step()

        print(f"Epoch [{epoch:02d}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    print(f"\n{name} 최종 테스트 정확도: {test_acc:.2f}%")
    return model


def evaluate_model_from_checkpoint(model, model_path, test_loader, name):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"{name} 모델 로드 완료: {model_path}")
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        print(f"{name} 평가 결과 -> Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        return model, True
    return model, False

if __name__ == "__main__":

    print("\n[1] MNIST 데이터 로드 중...")
    mnist_train_loader, mnist_test_loader = get_mnist_loaders(batch_size=128)

    mnist_model = MnistCNN()
    mnist_loaded = False
    if os.path.exists("mnist_model.pth"):
        mnist_model.load_state_dict(torch.load("mnist_model.pth", map_location=DEVICE))
        mnist_model = mnist_model.to(DEVICE)
        print("MNIST 모델 로드 완료: mnist_model.pth")
        mnist_loaded = True
    else:
        mnist_model = train_model(
            mnist_model,
            mnist_train_loader,
            mnist_test_loader,
            epochs=10,
            lr=0.001,
            name="MNIST CNN"
        )
        torch.save(mnist_model.state_dict(), "mnist_model.pth")
        print("MNIST 모델 저장 완료: mnist_model.pth")

    mnist_test_loss, mnist_test_acc = evaluate(mnist_model, mnist_test_loader, nn.CrossEntropyLoss())
    print(f"MNIST 모델 {'로드 및 평가' if mnist_loaded else '학습 후 평가'} 완료 -> Test Loss: {mnist_test_loss:.4f} | Test Acc: {mnist_test_acc:.2f}%")

    print("\n[2] CIFAR-10 데이터 로드 중...")
    cifar_train_loader, cifar_test_loader = get_cifar_loaders(batch_size=128)

    cifar_model = CifarCNN()
    cifar_loaded = False
    cifar_trained = False

    if os.path.exists("cifar_model.pth"):
        cifar_model.load_state_dict(torch.load("cifar_model.pth", map_location=DEVICE))
        cifar_model = cifar_model.to(DEVICE)
        print("CIFAR-10 모델 로드 완료: cifar_model.pth")
        cifar_loaded = True
        cifar_test_loss, cifar_test_acc = evaluate(cifar_model, cifar_test_loader, nn.CrossEntropyLoss())
        print(f"로드된 CIFAR-10 정확도: {cifar_test_acc:.2f}%")

        if cifar_test_acc < 80.0:
            print("CIFAR-10 정확도 <80%이므로 모델을 재학습합니다.")
            cifar_loaded = False
            cifar_model = train_model(
                cifar_model,
                cifar_train_loader,
                cifar_test_loader,
                epochs=30,
                lr=0.1,
                name="CIFAR-10 CNN",
                optimizer_type='sgd'
            )
            cifar_trained = True
            torch.save(cifar_model.state_dict(), "cifar_model.pth")
            print("CIFAR-10 모델 저장 완료: cifar_model.pth")
    else:
        cifar_model = train_model(
            cifar_model,
            cifar_train_loader,
            cifar_test_loader,
            epochs=30,
            lr=0.1,
            name="CIFAR-10 CNN",
            optimizer_type='sgd'
        )
        cifar_trained = True
        torch.save(cifar_model.state_dict(), "cifar_model.pth")
        print("CIFAR-10 모델 저장 완료: cifar_model.pth")

    cifar_test_loss, cifar_test_acc = evaluate(cifar_model, cifar_test_loader, nn.CrossEntropyLoss())
    status = "재학습 후 평가" if cifar_trained else "로드 및 평가" if cifar_loaded else "학습 후 평가"
    print(f"CIFAR-10 모델 {status} 완료 -> Test Loss: {cifar_test_loss:.4f} | Test Acc: {cifar_test_acc:.2f}%")