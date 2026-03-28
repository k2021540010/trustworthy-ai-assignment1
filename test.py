import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


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
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_mnist_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor()
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
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor()
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

def fgsm_targeted(model, x, target, eps):
    model.eval()

    x_adv = x.clone().detach().requires_grad_(True)

    output = model(x_adv)

    target_tensor = torch.tensor([target], device=DEVICE) if x_adv.dim() == 3 else \
                    torch.full((x_adv.size(0),), target, device=DEVICE)
    loss = nn.CrossEntropyLoss()(output, target_tensor)

    model.zero_grad()
    loss.backward()

    x_adv = x_adv - eps * x_adv.grad.sign()

    x_adv = x_adv.detach().clamp(0, 1)

    return x_adv

def fgsm_untargeted(model, x, label, eps):
    model.eval()

    x_adv = x.clone().detach().requires_grad_(True)

    output = model(x_adv)

    label_tensor = torch.tensor([label], device=DEVICE) if x_adv.dim() == 3 else \
                   torch.full((x_adv.size(0),), label, device=DEVICE)
    loss = nn.CrossEntropyLoss()(output, label_tensor)

    model.zero_grad()
    loss.backward()

    x_adv = x_adv + eps * x_adv.grad.sign()

    x_adv = x_adv.detach().clamp(0, 1)

    return x_adv

def pgd_targeted(model, x, target, k, eps, eps_step):
    model.eval()

    x_adv = x.clone().detach()

    target_tensor = torch.tensor([target], device=DEVICE) if x.dim() == 3 else \
                    torch.full((x.size(0),), target, device=DEVICE)

    for i in range(k):
        x_adv = x_adv.requires_grad_(True)

        output = model(x_adv)

        loss = nn.CrossEntropyLoss()(output, target_tensor)

        model.zero_grad()
        loss.backward()

        x_adv = x_adv - eps_step * x_adv.grad.sign()

        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)

        x_adv = x_adv.detach().clamp(0, 1)

    return x_adv


def pgd_untargeted(model, x, label, k, eps, eps_step):
    model.eval()

    x_adv = x.clone().detach()

    label_tensor = torch.tensor([label], device=DEVICE) if x.dim() == 3 else \
                   torch.full((x.size(0),), label, device=DEVICE)

    for i in range(k):
        x_adv = x_adv.requires_grad_(True)

        output = model(x_adv)

        loss = nn.CrossEntropyLoss()(output, label_tensor)

        model.zero_grad()
        loss.backward()

        x_adv = x_adv + eps_step * x_adv.grad.sign()

        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)

        x_adv = x_adv.detach().clamp(0, 1)

    return x_adv

os.makedirs('results', exist_ok=True)

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

def evaluate_attack(model, test_loader, attack_fn, attack_kwargs,
                    attack_type, n_samples=100, dataset_name="MNIST"):
    model.eval()
    success = 0
    total = 0

    for images, labels in test_loader:
        for i in range(images.size(0)):
            if total >= n_samples:
                break

            x = images[i].unsqueeze(0).to(DEVICE)  
            label = labels[i].item()

            if attack_type == 'targeted':
                target = (label + 1) % 10
                x_adv = attack_fn(model, x, target, **attack_kwargs)
                pred = model(x_adv).argmax(dim=1).item()
                if pred == target:
                    success += 1
            else:
                x_adv = attack_fn(model, x, label, **attack_kwargs)
                pred = model(x_adv).argmax(dim=1).item()
                if pred != label:
                    success += 1

            total += 1

        if total >= n_samples:
            break

    success_rate = 100. * success / total
    print(f"[{dataset_name}] {attack_fn.__name__} (eps={attack_kwargs.get('eps')}) "
          f"공격 성공률: {success_rate:.2f}% ({success}/{total})")
    return success_rate

def visualize_attack(model, test_loader, attack_fn, attack_kwargs,
                     attack_type, n_viz=5, dataset_name="MNIST", save_name="attack"):
    model.eval()
    samples = []  

    is_cifar = (dataset_name == "CIFAR-10")

    for images, labels in test_loader:
        for i in range(images.size(0)):
            if len(samples) >= n_viz:
                break

            x = images[i].unsqueeze(0).to(DEVICE)
            label = labels[i].item()

            if attack_type == 'targeted':
                target = (label + 1) % 10
                x_adv = attack_fn(model, x, target, **attack_kwargs)
            else:
                x_adv = attack_fn(model, x, label, **attack_kwargs)

            orig_pred = model(x).argmax(dim=1).item()
            adv_pred  = model(x_adv).argmax(dim=1).item()

            if orig_pred != label: 
                continue

            samples.append((
                x.cpu().squeeze(0),
                x_adv.cpu().squeeze(0),
                label, orig_pred, adv_pred
            ))

        if len(samples) >= n_viz:
            break

    if len(samples) == 0:
        print(f"[{dataset_name}] {save_name}: 공격 성공 샘플 없음, 시각화 건너뜀")
        return

    fig, axes = plt.subplots(len(samples), 3, figsize=(10, 3 * len(samples)))
    fig.suptitle(f"{dataset_name} - {attack_fn.__name__} (eps={attack_kwargs.get('eps')})",
                 fontsize=14, fontweight='bold')

    col_titles = ['Original', 'Adversarial', 'Perturbation (x10)']
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=12)

    for row, (x_orig, x_adv, true_label, orig_pred, adv_pred) in enumerate(samples):
        perturbation = (x_adv - x_orig) * 10 

        if is_cifar:
            x_orig_show = x_orig.clamp(0, 1).permute(1, 2, 0).numpy()
            x_adv_show  = x_adv.clamp(0, 1).permute(1, 2, 0).numpy()
            pert_show   = perturbation.clamp(0, 1).permute(1, 2, 0).numpy()
            cmap = None
            orig_label_str = CIFAR10_CLASSES[true_label]
            orig_pred_str  = CIFAR10_CLASSES[orig_pred]
            adv_pred_str   = CIFAR10_CLASSES[adv_pred]
        else:
            x_orig_show = x_orig.clamp(0, 1).squeeze().numpy()
            x_adv_show  = x_adv.clamp(0, 1).squeeze().numpy()
            pert_show   = perturbation.squeeze().numpy()
            cmap = 'gray'
            orig_label_str = str(true_label)
            orig_pred_str  = str(orig_pred)
            adv_pred_str   = str(adv_pred)

        axes[row][0].imshow(x_orig_show, cmap=cmap)
        axes[row][0].set_xlabel(f"True: {orig_label_str}\nPred: {orig_pred_str}", fontsize=9)

        axes[row][1].imshow(x_adv_show, cmap=cmap)
        axes[row][1].set_xlabel(f"Pred: {adv_pred_str}", fontsize=9, color='red')

        axes[row][2].imshow(pert_show, cmap=cmap)
        axes[row][2].set_xlabel("Perturbation x10", fontsize=9)

        for col in range(3):
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])

    plt.tight_layout()
    save_path = f"results/{save_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"시각화 저장 완료: {save_path}")

def run_all_attacks(mnist_model, cifar_model,
                    mnist_test_loader, cifar_test_loader):

    print("\n" + "="*60)
    print("  공격 실행 시작")
    print("="*60)

    mnist_eps  = 0.3
    cifar_eps  = 0.03  
    pgd_k      = 40
    pgd_step   = 0.01

    attacks = [
        (fgsm_targeted,   'targeted',   {'eps': mnist_eps},
         'MNIST',   mnist_model, mnist_test_loader, 'mnist_fgsm_targeted'),
        (fgsm_untargeted, 'untargeted', {'eps': mnist_eps},
         'MNIST',   mnist_model, mnist_test_loader, 'mnist_fgsm_untargeted'),
        (pgd_targeted,    'targeted',   {'eps': mnist_eps, 'k': pgd_k, 'eps_step': pgd_step},
         'MNIST',   mnist_model, mnist_test_loader, 'mnist_pgd_targeted'),
        (pgd_untargeted,  'untargeted', {'eps': mnist_eps, 'k': pgd_k, 'eps_step': pgd_step},
         'MNIST',   mnist_model, mnist_test_loader, 'mnist_pgd_untargeted'),

        (fgsm_targeted,   'targeted',   {'eps': cifar_eps},
         'CIFAR-10', cifar_model, cifar_test_loader, 'cifar_fgsm_targeted'),
        (fgsm_untargeted, 'untargeted', {'eps': cifar_eps},
         'CIFAR-10', cifar_model, cifar_test_loader, 'cifar_fgsm_untargeted'),
        (pgd_targeted,    'targeted',   {'eps': cifar_eps, 'k': pgd_k, 'eps_step': pgd_step/10},
         'CIFAR-10', cifar_model, cifar_test_loader, 'cifar_pgd_targeted'),
        (pgd_untargeted,  'untargeted', {'eps': cifar_eps, 'k': pgd_k, 'eps_step': pgd_step/10},
         'CIFAR-10', cifar_model, cifar_test_loader, 'cifar_pgd_untargeted'),
    ]

    results = {}
    for (attack_fn, attack_type, kwargs, dataset_name,
         model, loader, save_name) in attacks:

        rate = evaluate_attack(
            model, loader, attack_fn, kwargs,
            attack_type, n_samples=100, dataset_name=dataset_name
        )
        results[save_name] = rate

        visualize_attack(
            model, loader, attack_fn, kwargs,
            attack_type, n_viz=5,
            dataset_name=dataset_name, save_name=save_name
        )

    print("\n" + "="*60)
    print("  공격 성공률 요약")
    print("="*60)
    for name, rate in results.items():
        print(f"  {name:35s}: {rate:.2f}%")

    return results

def run_eps_analysis(mnist_model, cifar_model,
                     mnist_test_loader, cifar_test_loader):

    print("\n" + "="*60)
    print("  eps별 공격 성공률 분석")
    print("="*60)

    eps_list = [0.05, 0.1, 0.2, 0.3]
    pgd_k    = 40

    analysis = {
        'MNIST_fgsm_targeted':   {},
        'MNIST_fgsm_untargeted': {},
        'MNIST_pgd_targeted':    {},
        'MNIST_pgd_untargeted':  {},
        'CIFAR_fgsm_targeted':   {},
        'CIFAR_fgsm_untargeted': {},
        'CIFAR_pgd_targeted':    {},
        'CIFAR_pgd_untargeted':  {},
    }

    for eps in eps_list:
        print(f"\n--- eps = {eps} ---")
        eps_str = str(eps).replace('.', '')

        analysis['MNIST_fgsm_targeted'][eps] = evaluate_attack(
            mnist_model, mnist_test_loader, fgsm_targeted,
            {'eps': eps}, 'targeted', n_samples=100, dataset_name="MNIST")
        visualize_attack(
            mnist_model, mnist_test_loader, fgsm_targeted, {'eps': eps},
            'targeted', n_viz=5, dataset_name="MNIST",
            save_name=f"mnist_fgsm_targeted_eps{eps_str}")

        analysis['MNIST_fgsm_untargeted'][eps] = evaluate_attack(
            mnist_model, mnist_test_loader, fgsm_untargeted,
            {'eps': eps}, 'untargeted', n_samples=100, dataset_name="MNIST")
        visualize_attack(
            mnist_model, mnist_test_loader, fgsm_untargeted, {'eps': eps},
            'untargeted', n_viz=5, dataset_name="MNIST",
            save_name=f"mnist_fgsm_untargeted_eps{eps_str}")

        analysis['MNIST_pgd_targeted'][eps] = evaluate_attack(
            mnist_model, mnist_test_loader, pgd_targeted,
            {'eps': eps, 'k': pgd_k, 'eps_step': eps/10},
            'targeted', n_samples=100, dataset_name="MNIST")
        visualize_attack(
            mnist_model, mnist_test_loader, pgd_targeted,
            {'eps': eps, 'k': pgd_k, 'eps_step': eps/10},
            'targeted', n_viz=5, dataset_name="MNIST",
            save_name=f"mnist_pgd_targeted_eps{eps_str}")

        analysis['MNIST_pgd_untargeted'][eps] = evaluate_attack(
            mnist_model, mnist_test_loader, pgd_untargeted,
            {'eps': eps, 'k': pgd_k, 'eps_step': eps/10},
            'untargeted', n_samples=100, dataset_name="MNIST")
        visualize_attack(
            mnist_model, mnist_test_loader, pgd_untargeted,
            {'eps': eps, 'k': pgd_k, 'eps_step': eps/10},
            'untargeted', n_viz=5, dataset_name="MNIST",
            save_name=f"mnist_pgd_untargeted_eps{eps_str}")

        analysis['CIFAR_fgsm_targeted'][eps] = evaluate_attack(
            cifar_model, cifar_test_loader, fgsm_targeted,
            {'eps': eps}, 'targeted', n_samples=100, dataset_name="CIFAR-10")
        visualize_attack(
            cifar_model, cifar_test_loader, fgsm_targeted, {'eps': eps},
            'targeted', n_viz=5, dataset_name="CIFAR-10",
            save_name=f"cifar_fgsm_targeted_eps{eps_str}")

        analysis['CIFAR_fgsm_untargeted'][eps] = evaluate_attack(
            cifar_model, cifar_test_loader, fgsm_untargeted,
            {'eps': eps}, 'untargeted', n_samples=100, dataset_name="CIFAR-10")
        visualize_attack(
            cifar_model, cifar_test_loader, fgsm_untargeted, {'eps': eps},
            'untargeted', n_viz=5, dataset_name="CIFAR-10",
            save_name=f"cifar_fgsm_untargeted_eps{eps_str}")

        analysis['CIFAR_pgd_targeted'][eps] = evaluate_attack(
            cifar_model, cifar_test_loader, pgd_targeted,
            {'eps': eps, 'k': pgd_k, 'eps_step': eps/10},
            'targeted', n_samples=100, dataset_name="CIFAR-10")
        visualize_attack(
            cifar_model, cifar_test_loader, pgd_targeted,
            {'eps': eps, 'k': pgd_k, 'eps_step': eps/10},
            'targeted', n_viz=5, dataset_name="CIFAR-10",
            save_name=f"cifar_pgd_targeted_eps{eps_str}")

        analysis['CIFAR_pgd_untargeted'][eps] = evaluate_attack(
            cifar_model, cifar_test_loader, pgd_untargeted,
            {'eps': eps, 'k': pgd_k, 'eps_step': eps/10},
            'untargeted', n_samples=100, dataset_name="CIFAR-10")
        visualize_attack(
            cifar_model, cifar_test_loader, pgd_untargeted,
            {'eps': eps, 'k': pgd_k, 'eps_step': eps/10},
            'untargeted', n_viz=5, dataset_name="CIFAR-10",
            save_name=f"cifar_pgd_untargeted_eps{eps_str}")

    print("\n" + "="*60)
    print("  eps별 공격 성공률 요약표")
    print("="*60)
    print(f"{'Attack':<25} " + " ".join(f"eps={e:<5}" for e in eps_list))
    print("-"*60)
    for attack_name, eps_results in analysis.items():
        row = f"{attack_name:<25} "
        row += " ".join(f"{eps_results[e]:>8.1f}%" for e in eps_list)
        print(row)

    return analysis

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
    
    run_eps_analysis(mnist_model, cifar_model,
                     mnist_test_loader, cifar_test_loader)