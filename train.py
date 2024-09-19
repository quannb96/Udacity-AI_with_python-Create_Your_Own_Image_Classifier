import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.models import vgg16, VGG16_Weights
from collections import OrderedDict
import os

def get_input_args():
    parser = argparse.ArgumentParser(description="Train a new network on a dataset and save the model as a checkpoint.")
    parser.add_argument('data_dir', type=str, help='Directory of the dataset')
    parser.add_argument('--save_dir', type=str, default='.', help='Save checkpoint')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate as float (default: 0.001)')
    parser.add_argument('--hidden_units', type=int, default=4096, help='Hidden units as int (default: 4096)')
    parser.add_argument('--epochs', type=int, default=3, help='Epochs as int (default: 3)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    return parser.parse_args()
    
def get_data_loaders(data_dir):
    data_transforms = {
        'training': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        'training': datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms['training']),
        'validation': datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=data_transforms['validation'])
    }

    dataloaders = {
        'training': torch.utils.data.DataLoader(image_datasets['training'], batch_size=64, shuffle=True),
        'validation': torch.utils.data.DataLoader(image_datasets['validation'], batch_size=64, shuffle=True)
    }

    return dataloaders, image_datasets

def build_model(hidden_units):
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    return model

def train_model(model, dataloaders, criterion, optimizer, epochs, gpu):
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    steps = 0
    print_every = 15

    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in dataloaders['training']:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                test_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for inputs, labels in dataloaders['validation']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        output = model.forward(inputs)
                        test_loss += criterion(output, labels).item()
                        ps = torch.exp(output)
                        accuracy += (ps.max(dim=1)[1] == labels.data).float().mean().item()

                accuracy = accuracy / len(dataloaders['validation'])
                print(f'Epoch [{epoch+1}/{epochs}], '
                      f'Train Loss: {running_loss/print_every:.3f}, '
                      f'Test Loss: {test_loss/len(dataloaders["validation"]):.3f}, '
                      f'Test Accuracy: {accuracy:.3f}')
                running_loss = 0
                model.train()

def save_checkpoint(model, image_datasets, optimizer, epochs, save_dir):
    model.class_to_idx = image_datasets['training'].class_to_idx
    checkpoint = {
        'classifier': model.classifier,
        'learning_rate': optimizer.param_groups[0]['lr'],
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))

def main():
    args = get_input_args()
    dataloaders, image_datasets = get_data_loaders(args.data_dir)
    model = build_model(args.hidden_units)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    train_model(model, dataloaders, criterion, optimizer, args.epochs, args.gpu)
    save_checkpoint(model, image_datasets, optimizer, args.epochs, args.save_dir)

if __name__ == "__main__":
    main()