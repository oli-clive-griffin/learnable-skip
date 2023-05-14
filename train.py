import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, random_split
from model import SimpleLearnableSkipNet, SimpleSkipNet
from resnet.resnet import ResNet, LearnedSkipBottleneck, Bottleneck
import scipy.io
from tqdm import tqdm
import wandb

def get_cifar10():
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(
        root='./data/cifar10', train=True,
        download=True, transform=transform)

    train_loader = DataLoader(
        trainset, batch_size=128,
        shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data/cifar10', train=False,
        download=True, transform=transform)

    test_loader = DataLoader(
        testset, batch_size=128,
        shuffle=False, num_workers=2)

    return train_loader, test_loader

# def get_stanford_dogs(root="data/stanford_dogs", batch_size=128, num_workers=2):
#     transform = dogs_transform()
#     train_mat = scipy.io.loadmat(os.path.join(root, "lists", "train_list.mat"))
#     test_mat = scipy.io.loadmat(os.path.join(root, "lists", "test_list.mat"))

#     all_data = ImageFolder(os.path.join(root, "Images"), transform=transform)

#     train_indices = [idx for idx, (_, target) in enumerate(all_data.imgs) if target in train_mat["labels"]]
#     test_indices = [idx for idx, (_, target) in enumerate(all_data.imgs) if target in test_mat["labels"]]

#     train_set = Subset(all_data, train_indices)
#     test_set = Subset(all_data, test_indices)

#     train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#     return train_loader, test_loader, len(all_data.classes)

def dogs_transform():
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    return transform

def _old_get_stanford_dogs(root="data/stanford_dogs", batch_size=128, num_workers=2, num_classes=10):
    transform = dogs_transform()
    train_mat = scipy.io.loadmat(os.path.join(root, "lists", "train_list.mat"))
    test_mat = scipy.io.loadmat(os.path.join(root, "lists", "test_list.mat"))

    all_data = ImageFolder(os.path.join(root, "Images"), transform=transform)
    
    # Select a subset of classes
    selected_class_indices = list(range(num_classes))

    train_indices = [idx for idx, (_, target) in enumerate(all_data.imgs) if target in train_mat["labels"] and target - 1 in selected_class_indices]
    test_indices = [idx for idx, (_, target) in enumerate(all_data.imgs) if target in test_mat["labels"] and target - 1 in selected_class_indices]

    train_set = Subset(all_data, train_indices)
    test_set = Subset(all_data, test_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, num_classes


def get_stanford_dogs(root="data/stanford_dogs", batch_size=128, num_workers=2, num_classes=10, val_split=0.1):
    transform = dogs_transform()
    train_mat = scipy.io.loadmat(os.path.join(root, "lists", "train_list.mat"))
    test_mat = scipy.io.loadmat(os.path.join(root, "lists", "test_list.mat"))

    all_data = ImageFolder(os.path.join(root, "Images"), transform=transform)
    
    # Select a subset of classes
    selected_class_indices = list(range(num_classes))

    train_indices = [idx for idx, (_, target) in enumerate(all_data.imgs) if target in train_mat["labels"] and target - 1 in selected_class_indices]
    test_indices = [idx for idx, (_, target) in enumerate(all_data.imgs) if target in test_mat["labels"] and target - 1 in selected_class_indices]

    train_set = Subset(all_data, train_indices)
    test_set = Subset(all_data, test_indices)

    # Split the train_set into train and validation sets
    val_size = int(len(train_set) * val_split)
    train_size = len(train_set) - val_size
    train_set, val_set = random_split(train_set, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, num_classes


if __name__ == "__main__":
    while True:
        train_loader, val_loader, test_loader, num_classes = get_stanford_dogs(num_classes=10)
        device = torch.device("mps")

        num_epochs = 1

        models = [
            ResNet(block_constructor=LearnedSkipBottleneck, layers=[3, 4, 6, 3], num_classes=num_classes),
            ResNet(block_constructor=Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes),
        ]

        for model in models:
            wandb.init(project="Resnet learned skip", config={"model_name": model.name })
            model.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            for epoch in range(num_epochs):
                epoch_log_dict = {}
                if model.learned:
                    gates = model.get_gates()
                    if gates is not None:
                        for i, gate in enumerate(gates):
                            epoch_log_dict[f"gate_{i}"] = gate

                print(f"Epoch {epoch + 1} / {num_epochs}")
                running_loss = 0.0
                correct = 0
                total = 0

                model.train()
                for i, data in tqdm(enumerate(train_loader, 0)):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()

                    outputs = model(inputs)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                train_accuracy = correct / total
                train_loss = running_loss / (i + 1)

                model.eval()
                running_loss_val = 0.0
                correct_val = 0
                total_val = 0
                with torch.no_grad():
                    for i, data in tqdm(enumerate(val_loader, 0)):
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)

                        outputs = model(inputs)
                        loss_val = criterion(outputs, labels)

                        running_loss_val += loss_val.item()

                        _, predicted_val = torch.max(outputs.data, 1)
                        total_val += labels.size(0)
                        correct_val += (predicted_val == labels).sum().item()

                val_accuracy = correct_val / total_val
                val_loss = running_loss_val / (i + 1)

                epoch_log_dict["Train Loss"] = train_loss,
                epoch_log_dict["Train Accuracy"] = train_accuracy,
                epoch_log_dict["Validation Loss"] = val_loss,
                epoch_log_dict["Validation Accuracy"] = val_accuracy,
                epoch_log_dict["Epoch"] = epoch + 1,

                wandb.log(epoch_log_dict)

            # Test the model
            model.eval()
            correct_test = 0
            total_test = 0
            with torch.no_grad():
                for i, data in tqdm(enumerate(test_loader, 0)):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)

                    _, predicted_test = torch.max(outputs.data, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted_test == labels).sum().item()

            test_accuracy = correct_test / total_test
            wandb.log({"Test Accuracy": test_accuracy})

            wandb.finish()

