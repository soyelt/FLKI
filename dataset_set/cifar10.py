import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class cifar10:
    def __init__(
        self,
        data_path='./data/cifar10',
        num_clients=10,
        alpha=0.5,
        batch_size=64,
        test_batch_size=64
    ):
        self.num_clients = num_clients
        self.alpha = alpha
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size

        # ======================================================
        # CIFAR-10 transforms
        # ======================================================
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            ),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            ),
        ])

        # ======================================================
        # Load CIFAR-10 (只加载索引和标签，不占内存)
        # ======================================================
        train_dataset = datasets.CIFAR10(
            root=data_path,
            train=True,
            download=True,
            transform=transform_train
        )

        test_dataset = datasets.CIFAR10(
            root=data_path,
            train=False,
            download=True,
            transform=transform_test
        )

        # ======================================================
        # Dirichlet non-IID split（正确方式）
        # CIFAR10 用 targets 而不是 samples
        # ======================================================
        labels = np.array(train_dataset.targets)
        num_classes = 10

        idx_per_class = {
            c: np.where(labels == c)[0]
            for c in range(num_classes)
        }

        self.client_indices = {i: [] for i in range(self.num_clients)}

        for c in range(num_classes):
            idx = idx_per_class[c]
            np.random.shuffle(idx)

            proportions = np.random.dirichlet(
                np.ones(self.num_clients) * self.alpha
            )

            proportions = (np.cumsum(proportions) * len(idx)).astype(int)
            proportions[-1] = len(idx)

            split = np.split(idx, proportions[:-1])

            for client_id in range(self.num_clients):
                self.client_indices[client_id].extend(split[client_id].tolist())

        # ======================================================
        # 安全检查：无重复样本
        # ======================================================
        all_indices = sum(self.client_indices.values(), [])
        assert len(all_indices) == len(set(all_indices)), \
            "❌ Sample overlap detected in CIFAR-10 split!"

        # ======================================================
        # 构建每个客户端的 DataLoader（懒加载）
        # ======================================================
        self.tr_loaders = []
        total_count = 0

        for client_id in range(self.num_clients):
            indices = self.client_indices[client_id]
            subset = Subset(train_dataset, indices)

            loader = DataLoader(
                subset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )

            self.tr_loaders.append(loader)
            total_count += len(indices)

        print(f"[CIFAR-10] Total training samples distributed: {total_count}")

        # ======================================================
        # 全局测试集 DataLoader
        # ======================================================
        self.te_loader = DataLoader(
            test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    # ======================================================
    # 与 ImageNet / TinyImageNet 对齐的 API
    # ======================================================
    def get_tr_loaders(self):
        return self.tr_loaders

    def get_te_loader(self):
        return self.te_loader

    def get_trainer_data(self):
        """
        返回每个客户端的样本索引（而不是数据本身）
        """
        return self.client_indices

    def save_trainer_data(self):
        """
        返回每个客户端的数据量
        """
        return [len(self.client_indices[cid]) for cid in range(self.num_clients)]
