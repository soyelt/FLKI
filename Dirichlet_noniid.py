from args import args
import pickle
from collections import defaultdict
import os
import random
import numpy as np
import torch

from torch.utils.data import Subset, DataLoader

def get_train(dataset, indices, batch_size):
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )



def sample_dirichlet_train_data_train(dataset, num_clients, alpha):
    """
    Partition dataset using Dirichlet distribution for non-IID simulation.
    
    Args:
        dataset: PyTorch dataset (with 'targets' or 'samples' attribute)
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter
    
    Returns:
        client_indices: Dict mapping client_id to sample indices
        client_label_counts: Dict mapping client_id to class distribution
    """
    # Extract labels from dataset
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'samples'):
        labels = np.array([lbl for _, lbl in dataset.samples])
    else:
        raise AttributeError("Dataset has neither 'targets' nor 'samples' attribute")
    
    num_classes = labels.max() + 1
    idx_per_class = {c: np.where(labels == c)[0] for c in range(num_classes)}

    client_indices = {i: [] for i in range(num_clients)}
    client_label_counts = {i: defaultdict(int) for i in range(num_clients)}

    for c in range(num_classes):
        idx_list = idx_per_class[c]
        np.random.shuffle(idx_list)

        # Sample from Dirichlet distribution
        props = np.random.dirichlet([alpha] * num_clients)
        props = (props / props.sum() * len(idx_list)).astype(int)

        start = 0
        for client_id, count in enumerate(props):
            selected = idx_list[start:start + count]
            client_indices[client_id].extend(selected)
            client_label_counts[client_id][c] += len(selected)
            start += count

    return client_indices, client_label_counts
