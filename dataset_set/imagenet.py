import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms


class imagenet:
    def __init__(self, data_path="data/imagenet", num_clients=10, alpha=0.5, batch_size=32, test_batch_size=64):
        self.data_path = data_path
        self.num_clients = num_clients
        self.alpha = alpha
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size

        # ======================================================
        #    ImageNet transforms
        # ======================================================
        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])

        self.transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])

        # ======================================================
        #    Only load path/label metadata — NO image loading
        # ======================================================
        print("Loading ImageNet ImageFolder metadata ...")

        self.train_dataset = datasets.ImageFolder(
            os.path.join(data_path, "train"),
            transform=self.transform_train
        )

        self.test_dataset = datasets.ImageFolder(
            os.path.join(data_path, "val"),
            transform=self.transform_test
        )

        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples:   {len(self.test_dataset)}")

        # ==================================================================
        #                    Dirichlet index-only split
        # ==================================================================
        print("Building Dirichlet partitions...")
        self.client_indices = self.build_dirichlet_partitions(
            alpha=self.alpha,
            num_clients=self.num_clients
        )

        # ==================================================================
        #       Build DataLoader for each client (same as tinyimagenet)
        # ==================================================================
        self.tr_loaders = []
        total_count = 0

        for cid in range(num_clients):
            indices = self.client_indices[cid]
            subset = Subset(self.train_dataset, indices)
            loader = DataLoader(
                subset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            self.tr_loaders.append(loader)
            total_count += len(indices)

        print(f"Total training samples distributed: {total_count}")

        # Test loader (global)
        self.te_loader = DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    # ======================================================================
    #   Dirichlet split (index-only, zero memory overhead)
    # ======================================================================
    def build_dirichlet_partitions(self, alpha, num_clients):
        labels = np.array([s[1] for s in self.train_dataset.samples])
        num_classes = labels.max() + 1

        idx_per_class = {c: np.where(labels == c)[0].tolist()
                         for c in range(num_classes)}

        client_indices = {i: [] for i in range(num_clients)}

        for c in range(num_classes):
            idx_list = idx_per_class[c]
            np.random.shuffle(idx_list)

            # Dirichlet proportions
            proportions = np.random.dirichlet(alpha * np.ones(num_clients))
            proportions = (np.cumsum(proportions) * len(idx_list)).astype(int)
            proportions[-1] = len(idx_list)

            split = np.split(idx_list, proportions[:-1])

            for client_id in range(num_clients):
                client_indices[client_id].extend(split[client_id])

        # 🔴 关键安全检查
        all_indices = sum(client_indices.values(), [])
        assert len(all_indices) == len(set(all_indices)), "Sample overlap detected!"

        return client_indices

    # ======================================================================
    #             ★★ Provide exactly same API as tinyimagenet ★★
    # ======================================================================
    def get_tr_loaders(self):
        return self.tr_loaders

    def get_te_loader(self):
        return self.te_loader
    # ======================================================================
    #   Return client indices instead of real data (to avoid huge memory use)
    # ======================================================================
    def get_trainer_data(self):
        """
        返回每个客户端的数据索引列表（不加载图像，不占内存）
        """
        return self.client_indices

    def save_trainer_data(self):
        """
        返回每个客户端的数据量
        """
        return [len(self.client_indices[cid]) for cid in range(self.num_clients)]
