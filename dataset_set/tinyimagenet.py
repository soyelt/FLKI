import torch
# from args import args
from torchvision import datasets, transforms
import torchvision
from Dirichlet_noniid import *
from torch.utils.data import Dataset
import glob
from PIL import Image
from typing import Any


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


def sample_dirichlet_indices_with_labels(dataset, num_clients, alpha):
    # ========= 1. 显式遍历 dataset 拿 label =========
    labels = []
    for i in range(len(dataset)):
        _, lbl = dataset[i]
        labels.append(lbl)
    labels = np.array(labels)

    num_classes = labels.max() + 1

    # ========= 2. 按类别收集索引 =========
    idx_per_class = {
        c: np.where(labels == c)[0].tolist()
        for c in range(num_classes)
    }

    client_indices = {i: [] for i in range(num_clients)}
    client_label_counts = {i: defaultdict(int) for i in range(num_clients)}

    # ========= 3. 标准 label-skew Dirichlet =========
    for c in range(num_classes):
        idx_list = idx_per_class[c]
        np.random.shuffle(idx_list)

        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        proportions = (np.cumsum(proportions) * len(idx_list)).astype(int)
        proportions[-1] = len(idx_list)

        splits = np.split(idx_list, proportions[:-1])

        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split)
            client_label_counts[client_id][c] += len(split)

    # ========= 4. 强一致性检查 =========
    all_indices = sum(client_indices.values(), [])
    assert len(all_indices) == len(set(all_indices)), \
        "❌ Dirichlet split error: duplicated samples"

    return client_indices, client_label_counts



class tinyimagenet:
    def __init__(self):

        args.output_size = 200

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

        client_indices, client_label_counts = sample_dirichlet_indices_with_labels(train_dataset, args.trainer_num,
                                                                                   args.non_iid_degree)

        self.tr_loaders = []
        tr_count = 0
        for pos, indices in client_indices.items():
            if len(indices) == 1 or len(indices) == 0:
                print(pos)
            tr_count += len(indices)
            self.tr_loaders.append(get_train(train_dataset, indices, args.bs))
        print("number of total training points:", tr_count)
        self.te_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

        self.trainer_indices = client_indices
        self.trainer_label_counts = client_label_counts

    def get_tr_loaders(self):
        return self.tr_loaders

    def get_te_loader(self):
        return self.te_loader

    def get_trainer_data(self):
        return self.trainer_indices, self.trainer_label_counts

    def save_trainer_data(self):
        return [len(v) for v in self.trainer_indices.values()]
