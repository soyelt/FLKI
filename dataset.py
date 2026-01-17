from torchvision import datasets, transforms
from torch.utils.data import Dataset
from args import args
from PIL import Image
from typing import Any
import glob
import os

class TrainTinyImageNet(Dataset):
    def __init__(self, root, id, transform=None) -> None:
        super().__init__()
        self.filenames = glob.glob(root + "\\train\*\*\*.JPEG")
        self.transform = transform
        self.id_dict = id

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: Any) -> Any:
        img_path = self.filenames[idx]
        image = Image.open(img_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        label = self.id_dict[img_path.split('\\')[-3]]
        if self.transform:
            image = self.transform(image)
        return image, label


class ValTinyImageNet(Dataset):
    def __init__(self, root, id, transform=None):
        self.filenames = glob.glob(root + "\\val\images\*.JPEG")
        self.transform = transform
        self.id_dict = id
        self.cls_dic = {}
        for i, line in enumerate(open(root + '\\val\\val_annotations.txt', 'r')):
            a = line.split('\t')
            img, cls_id = a[0], a[1]
            self.cls_dic[img] = self.id_dict[cls_id]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = Image.open(img_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        label = self.cls_dic[img_path.split('\\')[-1]]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_dataset():
    data_path = './data'
    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10(data_path, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(data_path, train=False, download=True, transform=transform_test)

    elif args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        ])

        train_dataset = datasets.CIFAR100(data_path, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(data_path, train=False, download=True, transform=transform_test)

    elif args.dataset == 'mnist':
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])

        train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform_train)
        test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=transform_test)

    elif args.dataset == 'fmnist':
        transform_train = transforms.Compose([
            transforms.Resize(32),  # 将图像调整为32x32
            transforms.ToTensor()  # 将图像转换为Tensor
        ])

        transform_test = transforms.Compose([
            transforms.Resize(32),  # 将图像调整为32x32
            transforms.ToTensor(),  # 将图像转换为Tensor
            transforms.Normalize((0.2860,), (0.3530,))  # 归一化
        ])

        train_dataset = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform_train)
        test_dataset = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform_test)

    elif args.dataset == 'tinyimagenet':
        root = './data/tiny-imagenet-200'
        id_dic = {}
        for i, line in enumerate(open(root + '\wnids.txt', 'r')):
            id_dic[line.replace('\n', '')] = i
        data_transform = {
            "train": transforms.Compose([transforms.Resize(224),
                                         transforms.RandomCrop(224, padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "val": transforms.Compose([transforms.Resize(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
        train_dataset = TrainTinyImageNet(root, id=id_dic, transform=data_transform["train"])
        test_dataset = ValTinyImageNet(root, id=id_dic, transform=data_transform["val"])

    elif args.dataset == 'imagenet':
        root = './data/imagenet'
        data_transform = {
            "train": transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
            "val": transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        }

        train_dataset = datasets.ImageFolder(root=f'{root}/train', transform=data_transform["train"])
        test_dataset = datasets.ImageFolder(root=f'{root}/val', transform=data_transform["val"])

    return train_dataset, test_dataset



