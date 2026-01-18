from args import args
from torchvision import datasets, transforms
import torch
import os
from Dirichlet_noniid import *


class fmnist:
    def __init__(self):
        args.output_size = 10

        transform_train = transforms.Compose([
            transforms.Resize(32),  # 将图像调整为32x32
            transforms.ToTensor()   # 转换为Tensor
        ])
        transform_test = transforms.Compose([
            transforms.Resize(32),  # 将图像调整为32x32
            transforms.ToTensor(),  # 转换为Tensor
            transforms.Normalize((0.2860,), (0.3530,))  # 归一化
        ])

        # 下载和加载 FMNIST 数据集
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)

        # 使用 Dirichlet 采样数据
        tr_per_participant_list, tr_diversity = sample_dirichlet_train_data_train(train_dataset, args.trainer_num,
                                                                                  alpha=args.non_iid_degree
                                                                                )

        self.tr_loaders = []
        tr_count = 0
        for pos, indices in tr_per_participant_list.items():
            if len(indices) == 1 or len(indices) == 0:
                print(pos)
            tr_count += len(indices)
            batch_size = args.batch_size
            self.tr_loaders.append(get_train(train_dataset, indices, args.batch_size))
        print("number of total training points:", tr_count)
        self.te_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    def get_tr_loaders(self):
        return self.tr_loaders

    def get_te_loader(self):
        return self.te_loader

    def save_trainer_data(self):
        # 确保保存目录存在
        os.makedirs('trainers_data', exist_ok=True)
        data_num_dict = dict()
        # 保存每个训练者的数据和标签
        for idx, loader in enumerate(self.tr_loaders):
            trainer_data = []
            trainer_labels = []
            for inputs, labels in loader:
                trainer_data.append(inputs)
                trainer_labels.append(labels)

            # 将数据和标签合并并保存为 .pt 文件
            trainer_data = torch.cat(trainer_data)  # 合并所有批次的数据
            trainer_labels = torch.cat(trainer_labels)  # 合并所有批次的标签

            # 保存为 .pt 文件
            torch.save(trainer_data, os.path.join('trainers_data', f'trainer_{idx}_data_{args.dataset}_{args.non_iid_degree}.pt'))
            torch.save(trainer_labels, os.path.join('trainers_data', f'trainer_{idx}_labels_{args.dataset}_{args.non_iid_degree}.pt'))

            print(f"Trainer {idx} 总数据量: {len(trainer_data)} ")
            data_num_dict[idx] = len(trainer_data)
        return data_num_dict