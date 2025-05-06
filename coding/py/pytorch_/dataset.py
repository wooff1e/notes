import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode


def get_loader(img_size, batch_size, paths, train):
    transforms = get_train_transforms(img_size) if train else get_test_transforms(img_size)
    ds = CustomDataset(paths, transforms)
    kwargs = {'num_workers': 8, 'pin_memory': True} if (torch.cuda.is_available()) else {}
    dataloader = DataLoader(ds, batch_size, shuffle=train, **kwargs)
    return dataloader    


class CustomDataset(Dataset):
    def __init__(self, paths, transforms = transforms.ToTensor()):
        super().__init__()
        self.paths = paths
        self.transforms = transforms
        
    def __getitem__(self, index):
        img_path = str(self.paths[index])
        img = read_image(img_path, ImageReadMode.RGB_ALPHA)  # reads into uint8 tensor
        img = self.transforms(img)
        return img
        
    def __len__(self):
        return len(self.paths)


def get_train_transforms(img_size):
    transforms = transforms.Compose([
        transforms.RandomRotation(10),      # rotate +/- 10 degrees
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)], p=0.1),
        transforms.RandomGrayscale(p=0.075),
        transforms.Resize(img_size),        # resize shortest side
        transforms.CenterCrop(40),          # crop longest side    
        transforms.ToTensor(),
        transforms.Resize(img_size),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transforms

def get_test_transforms(img_size):
    transforms = transforms.Compose([
        transforms.Resize(img_size),        # resize shortest side
        transforms.CenterCrop(40),          # crop longest side    
        transforms.ToTensor(),
        transforms.Resize(img_size),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transforms

# turn the integer into a one-hot encoded tensor
target_transforms = transforms.Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))