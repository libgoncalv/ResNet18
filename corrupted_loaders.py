import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from corrupted_cifar import CustomCIFAR10


def create_loaders(hparam, corruption, hyper=False, root_dir='data/'):
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    train_transform = transforms.Compose([transforms.ToTensor(), normalize])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    
    # Train set
    trainset = CustomCIFAR10(corruption, root=root_dir, train=True, download=True, transform=train_transform)
    num_train = int(np.floor((1-hparam['percent_valid']) * len(trainset)))

    trainset.data = trainset.data[:num_train, :, :, :]
    trainset.targets = trainset.targets[:num_train]
    
    # Validation set
    valset = CustomCIFAR10(corruption, root=root_dir, train=True, download=True, transform=train_transform)
    valset.data = valset.data[num_train:, :, :, :]
    valset.targets = valset.targets[num_train:]
    
    # Test set
    testset = CustomCIFAR10(corruption, root=root_dir, train=False, download=True, transform=test_transform)


    train_loader = DataLoader(dataset=trainset, batch_size=hparam['batch_size'], shuffle=True, pin_memory=True, num_workers=16)
    valid_loader = DataLoader(dataset=valset, batch_size=hparam['batch_size'], shuffle=True, pin_memory=True, num_workers=16)
    test_loader = DataLoader(dataset=testset, batch_size=hparam['batch_size'], pin_memory=True, num_workers=16)

    return train_loader, valid_loader, test_loader
