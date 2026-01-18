import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from args import args


class mnist:
    def __init__(self):

        args.output_size = 10
        self.num_clients = args.trainer_num
        self.alpha = args.non_iid_degree

        # ======================================================
        # MNIST transforms
        # ======================================================
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor()
        ])

        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # ======================================================
        # Load MNIST（不加载进内存）
        # ======================================================
        train_dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform_train
        )

        test_dataset = datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform_test
        )

        # ======================================================
        # Dirichlet non-IID split（正确 & 省内存）
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
        # 安全检查
        # ======================================================
        all_indices = sum(self.client_indices.values(), [])
        assert len(all_indices) == len(set(all_indices)), \
            "❌ Sample overlap detected in MNIST split!"

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
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True
            )

            self.tr_loaders.append(loader)
            total_count += len(indices)

        print(f"[MNIST] Total training samples distributed: {total_count}")

        # ======================================================
        # 全局测试集
        # ======================================================
        self.te_loader = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

    # ======================================================
    # 统一 API（与你 CIFAR / ImageNet 完全一致）
    # ======================================================
    def get_tr_loaders(self):
        return self.tr_loaders

    def get_te_loader(self):
        return self.te_loader

    def get_trainer_data(self):
        """
        返回每个客户端的样本索引
        """
        return self.client_indices

    def save_trainer_data(self):
        """
        返回每个客户端的数据量（用于 FedAvg 聚合）
        """
        return [len(self.client_indices[cid]) for cid in range(self.num_clients)]
