from config import cfg
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def create_dataset_pytorch_cifar10(data_path, is_train=True, n_workers=8):


  if is_train:
      transform = transforms.Compose([
        transforms.RandomResizedCrop(cfg.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
  else:
      transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

  ds_cifar10 = torchvision.datasets.CIFAR10(root=data_path, train=is_train,
                                          download=False, transform=transform)
  data_loader = DataLoader(dataset=ds_cifar10, 
                          batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=n_workers)
  return data_loader


def create_dataset_pytorch(data_path):
  transform = transforms.Compose([
    transforms.Resize((cfg.image_height, cfg.image_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

  dataset = torchvision.datasets.ImageFloder(root=data_path, transform=transform)
  data_loader = DataLoader(dataset=dataset, 
                          batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=4)

  return data_loader
