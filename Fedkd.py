import os
from torch.utils.data import Dataset, DataLoader, TensorDataset
import dataset_set
import time
import torch.utils
import argparse
import copy
import model_2
import matplotlib.pyplot as plt
import dataset
import sever
from trainer_kd import Trainer
import torch
from torch.utils.data import Subset
from collections import defaultdict
import torch.nn.functional as F
import random
import torch.nn as nn
import pandas as pd
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def args_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', help="Dataset name")
    parser.add_argument('--num_classes', type=int, default=10, help="Number of classes")
    parser.add_argument('--agent', type=int, default=1, help='Number of agents')
    parser.add_argument('--trainer_num', type=int, default=8, help='Number of trainers')
    parser.add_argument("--non_iid_degree", type=float, default=1.0, help="Non-IID degree for Dirichlet distribution")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum")
    parser.add_argument('--bs', type=int, default=64, help="Batch size")
    parser.add_argument('--round', type=int, default=200, help="Communication rounds")
    parser.add_argument('--labda', type=float, default=1.0, help="Lambda parameter")
    parser.add_argument('--layer_coverage_tau', type=float, default=0.8, help="Layer coverage threshold for adaptive selection")
    return parser.parse_args()

def aggregate_model(w, data_num):
    """Aggregate models using weighted averaging."""
    global_model = copy.deepcopy(w[0])
    for k in global_model.keys():
        global_model[k] = torch.zeros_like(w[0][k]).float()

    total_data = sum(data_num)
    for i in range(len(w)):
        for k in global_model.keys():
            global_model[k] += w[i][k] * (data_num[i] / total_data)

    return global_model


class PseudoFeatureDataset(Dataset):
    """Dataset for pseudo features with optional soft labels."""
    def __init__(self, features, labels, logits=None, use_soft_labels=False):
        self.features = features
        self.labels = labels
        self.logits = logits
        self.use_soft_labels = use_soft_labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.logits[idx] if self.use_soft_labels and self.logits is not None else self.labels[idx]
        return x, y

def aggregate_gaussian_barycenter(mu_dict, Sigma_dict, data_size_dict):
    """
    Aggregate Gaussian distributions using Wasserstein barycenter.
    
    Args:
        mu_dict: Dict mapping client_id to mean vector (shape [D])
        Sigma_dict: Dict mapping client_id to covariance matrix (shape [D, D])
        data_size_dict: Dict mapping client_id to data size
    
    Returns:
        mu_agg: Aggregated mean vector
        Sigma_agg: Aggregated covariance matrix
    """
    device = next(iter(mu_dict.values())).device
    client_ids = mu_dict.keys()
    total_size = sum(data_size_dict[i] for i in client_ids)

    mu_agg = sum((data_size_dict[i] / total_size) * mu_dict[i] for i in client_ids)

    Sigma_agg = torch.zeros_like(next(iter(Sigma_dict.values()))).to(device)
    for i in client_ids:
        w_k = data_size_dict[i] / total_size
        mu_diff = (mu_dict[i] - mu_agg).unsqueeze(1)
        Sigma_k_term = Sigma_dict[i] + mu_diff @ mu_diff.T
        Sigma_agg += w_k * Sigma_k_term

    return mu_agg, Sigma_agg

def get_candidate_layers(dataset_name):
    # Get candidate layers for adaptive selection based on dataset/model.
    if dataset_name in ['mnist', 'fmnist']:
        # LeNet5: 5 layers
        return ['features.0', 'features.3', 'classifier.0', 'classifier.2', 'classifier.4']
    elif dataset_name == 'cifar10':
        # ResNet18: fine-grained layers
        layers = []
        layers.extend(['model0.0', 'model0.1', 'model0.3'])
        
        for i in [1, 2, 3, 4, 5, 6, 7, 8]:
            layers.extend([f'model{i}.0', f'model{i}.1', f'model{i}.3', f'model{i}.4'])
        
        for en in ['en1', 'en2', 'en3']:
            layers.extend([f'{en}.0', f'{en}.1'])
        
        layers.extend(['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8'])
        layers.extend(['aap', 'flatten', 'fc'])
        
        return layers
    elif dataset_name in ['tinyimagenet', 'imagenet']:
        # ViT-B16: transformer blocks
        layers = ['patch_embed']
        
        for i in range(12):
            layers.extend([f'blocks.{i}.norm1', f'blocks.{i}.attn'])
        
        layers.extend(['norm', 'head'])
        
        return layers
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def compute_layer_distinguishability(layer_mu_dict, layer_Sigma_dict, client_ids):
    """
    Calculate the distinguishability function S(l) for each layer
    
    S(l) = E_k[||μ_k^(l) - μ_global^(l)||^2] / E_k[Tr(Σ_k^(l))]
    
    Args:
        layer_mu_dict: dict {layer_name: {client_id: mu_tensor}}
        layer_Sigma_dict: dict {layer_name: {client_id: Sigma_tensor}}
        client_ids: list of client IDs
    
    Returns:
        S_dict: dict {layer_name: S(l) value}
    """
    S_dict = {}
    
    for layer_name in layer_mu_dict.keys():
        # Calculate global mean
        mu_global = torch.stack([layer_mu_dict[layer_name][k] for k in client_ids]).mean(dim=0)
        
        # Calculate numerator: E_k[||μ_k - μ_global||^2]
        numerator = 0.0
        for k in client_ids:
            mu_k = layer_mu_dict[layer_name][k]
            numerator += torch.norm(mu_k - mu_global, p=2) ** 2
        numerator /= len(client_ids)
        
        # Calculate denominator: E_k[Tr(Σ_k)]
        denominator = 0.0
        for k in client_ids:
            Sigma_k = layer_Sigma_dict[layer_name][k]
            denominator += torch.trace(Sigma_k)
        denominator /= len(client_ids)
        
        # Avoid division by zero
        if denominator > 1e-8:
            S_dict[layer_name] = (numerator / denominator).item()
        else:
            S_dict[layer_name] = 0.0
    
    return S_dict

def select_adaptive_layers(S_dict, tau=0.8):
    """
    Select personalized layers based on cumulative information coverage strategy
    
    K* = min{k | Σ(i=1 to k) S(l_i) / Σ(i=1 to L) S(i) >= tau}
    
    Args:
        S_dict: dict {layer_name: S(l) value}
        tau: coverage threshold (0, 1)
    
    Returns:
        selected_layers: list of selected layer names (top K* layers)
    """
    # Sort by S(l) in descending order
    sorted_layers = sorted(S_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate total information
    total_S = sum(S_dict.values())
    
    if total_S < 1e-8:
        # If all layers have S close to 0, return the layer with the highest S value
        return [sorted_layers[0][0]] if sorted_layers else []
    
    # Cumulative information coverage
    cumulative_S = 0.0
    selected_layers = []
    
    for layer_name, S_value in sorted_layers:
        cumulative_S += S_value
        selected_layers.append(layer_name)
        
        # Check if the coverage threshold is reached
        if cumulative_S / total_S >= tau:
            break
    
    return selected_layers

def subset_by_class_fraction(dataset, fraction_per_class, seed=42):

    random.seed(seed)
    class_indices = defaultdict(list)

    if hasattr(dataset, "targets"):
        labels = dataset.targets
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()
    elif hasattr(dataset, "filenames") and hasattr(dataset, "id_dict"):
        # For TinyImageNet, extract labels directly from file paths to avoid loading images
        labels = []
        for idx, img_path in enumerate(dataset.filenames):
            if hasattr(dataset, 'cls_dic'):  # ValTinyImageNet
                filename = img_path.split('\\')[-1]
                labels.append(dataset.cls_dic[filename])
            else:  # TrainTinyImageNet
                class_id = img_path.split('\\')[-3]
                labels.append(dataset.id_dict[class_id])
    else:
        labels = [dataset[i][1] for i in range(len(dataset))]

    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    selected_indices = []
    for label, indices in class_indices.items():
        k = max(1, int(len(indices) * fraction_per_class))
        sampled = random.sample(indices, k)
        selected_indices.extend(sampled)

    return Subset(dataset, selected_indices)

def total_loss_fn(model, device, public_loader, mu_agg, Sigma_agg, dataset_name, alpha_fdma, lambda_fdma):
    model.train()
    model = model.to(device)
    ce_loss_fn = nn.CrossEntropyLoss()

    all_features = []
    total_ce_loss = 0.0
    total_samples = 0

    extracted_features = []

    def hook_fn(module, input, output):
        if output.dim() == 4 and output.shape[2:] == (1, 1):
            output = output.squeeze(-1).squeeze(-1)
        extracted_features.append(output)

    if dataset_name == 'fmnist':
        handle = model.classifier[2].register_forward_hook(hook_fn)
    elif dataset_name in ['cifar10']:
        handle = model.fc.register_forward_hook(hook_fn)
    elif dataset_name in ['tinyimagenet']:
        handle = model.blocks[-1].register_forward_hook(hook_fn)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    for data, target in public_loader:
        data, target = data.to(device), target.to(device)
        extracted_features.clear()

        output = model(data)
        ce_loss = ce_loss_fn(output, target)
        total_ce_loss += ce_loss * data.size(0)  # Do not use item()
        total_samples += data.size(0)

        feature_batch = extracted_features[0]
        all_features.append(feature_batch)

    handle.remove()

    H = torch.cat(all_features, dim=0)  # [N, D]
    mu_pub = H.mean(dim=0, keepdim=True)
    centered = H - mu_pub
    Sigma_pub = centered.T @ centered / (H.size(0) - 1)

    mean_align = torch.norm(mu_pub.squeeze(0) - mu_agg.to(mu_pub.device), p=2) ** 2
    cov_diff = Sigma_pub - Sigma_agg.to(Sigma_pub.device)
    frobenius_loss = torch.norm(cov_diff, p='fro') ** 2
    fdma_loss = mean_align + lambda_fdma * frobenius_loss

    sup_loss = total_ce_loss / total_samples
    total_loss = sup_loss + alpha_fdma * fdma_loss

    return total_loss




def update(state_dict, device, args, train_loader):
    if args.dataset in ['mnist', 'fmnist']:
        local_model = model_2.LeNet5(args.num_classes)
    elif args.dataset in ['cifar10']:
        local_model = model_2.Resnet18(args.num_classes)
    elif args.dataset in ['tinyimagenet', 'imagenet']:
        local_model = model_2.ViT_B16()
    elif args.dataset in ['cifar100']:
        local_model = model_2.VGG11(args.num_classes)
    else:
        raise ValueError("Unsupported dataset")

    local_model.load_state_dict(copy.deepcopy(state_dict))
    local_model.to(device)
    local_model.train()

    optimizer = torch.optim.SGD(local_model.parameters(), lr=args.lr, momentum=args.momentum)
    Loss = torch.nn.CrossEntropyLoss()

    train_loss = 0.0
    correct = 0
    data_size = 0

    for _ in range(1):
        for batch_id, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = local_model(data)
            loss = Loss(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            data_size += len(data)

    train_loss /= data_size
    accuracy = 100.0 * correct / data_size

    S = sever.Sever(args, dataset_test, local_model.state_dict())
    total_acc = S.evaluate()

    return local_model.state_dict(), total_acc


def distill_train_and_evaluate(state_dict, device, public_loader, pseudo_dataset, dataset_name,
                               alpha, beta, gamma, temperature, steps):

    # Initialize the global model
    if args.dataset in ['mnist', 'fmnist']:
        model = model_2.LeNet5(args.num_classes)
    elif args.dataset in ['cifar10']:
        model = model_2.Resnet18(args.num_classes)
    elif args.dataset in ['tinyimagenet', 'imagenet']:
        model = model_2.ViT_B16()
    elif args.dataset in ['cifar100']:
        model = model_2.VGG11(args.num_classes)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    model.load_state_dict(copy.deepcopy(state_dict))
    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    ce_loss_fn = nn.CrossEntropyLoss()
    kl_loss_fn = nn.KLDivLoss(reduction='batchmean')

    # Check feature dimension
    sample_feature, _ = pseudo_dataset[0]
    feature_dim = sample_feature.shape[0]
    
    # Extract the classifier (latter part) module of the model
    if dataset_name == 'fmnist':
        original_classifier = model.classifier[3:]
        expected_dim = 84  # LeNet5 classifier[2] output dimension
    elif dataset_name == 'cifar10':
        original_classifier = model.fc
        expected_dim = 512  # ResNet18 fc input dimension
    elif dataset_name in ['tinyimagenet', 'imagenet']:
        original_classifier = model.head
        expected_dim = 768  # ViT-B16 head input dimension
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # If the feature dimension does not match the expected one, create a new classifier
    if feature_dim != expected_dim:
        classifier = nn.Linear(feature_dim, args.num_classes).to(device)
        # Recreate the optimizer to include the new classifier parameters
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        classifier = original_classifier.to(device)

    pseudo_loader = torch.utils.data.DataLoader(pseudo_dataset, batch_size=args.bs, shuffle=True)

    for step in range(steps):
        total_loss = 0

        # ========== Soft Label  ==========
        for feature, soft_label in pseudo_loader:
            feature = feature.to(device)              # shape: [B, D]
            soft_label = soft_label.to(device)        # shape: [B, C],  softmax/logits

            optimizer.zero_grad()

            student_output = classifier(feature) / temperature
            with torch.no_grad():
                target_soft = soft_label / temperature

            soft_loss = kl_loss_fn(
                F.log_softmax(student_output, dim=1),
                F.softmax(target_soft, dim=1)
            ) * (temperature ** 2)

            soft_loss.backward()
            optimizer.step()

            # total_loss += alpha * soft_loss.item()

        # ========== Hard Label ==========
        for feature, soft_label in pseudo_loader:
            feature = feature.to(device)
            soft_label = soft_label.to(device)

            hard_label = torch.argmax(soft_label, dim=1)

            optimizer.zero_grad()
            pred = classifier(feature)
            hard_loss = ce_loss_fn(pred, hard_label)
            hard_loss.backward()
            optimizer.step()

            # total_loss += beta * hard_loss.item()



    model.eval()
    S = sever.Sever(args, dataset_test, model.state_dict())
    total_acc = S.evaluate()

    return model.state_dict(), total_acc


def extract_fc_layer_weights_by_dataset(model, dataset_name):

    if dataset_name == 'fmnist':
        target_layer = model.classifier[2]
    elif dataset_name == 'cifar10':
        target_layer = model.fc
    elif dataset_name in ['tinyimagenet', 'imagenet']:
        target_layer = model.head  # ViT uses 'head' instead of 'fc'
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    weight_matrix = target_layer.weight.data.clone()  # shape: [n, d]
    return weight_matrix

def extract_layer_weights(model, dataset_name, layer_name):
    """
    Extract the weights of the specified layer according to the layer name
    For convolutional layers, return None (not applicable for OT)
    For linear layers, return the weight matrix
    """
    # Get the module according to the layer name
    parts = layer_name.split('.')
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    
    # Check the layer type
    if isinstance(module, torch.nn.Linear):
        return module.weight.data.clone()
    elif isinstance(module, (torch.nn.Conv2d, torch.nn.Sequential)):
        # Convolutional layers or Sequential not applicable for OT, return None
        return None
    elif hasattr(module, 'weight'):
        # Other layers with weight attribute
        weight = module.weight.data
        if weight.dim() == 2:  # Only process 2D weights (linear layers)
            return weight.clone()
        else:
            return None  # High-dimensional weights of convolutional layers not applicable for OT
    else:
        return None

def replace_single_layer_weights(model, personalized_weights, dataset_name, layer_name):
    """
    Replace weights for the specified layer (only for linear layers)
    """
    # Get the module according to the layer name
    parts = layer_name.split('.')
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    
    # Replace weights (only for linear layers)
    with torch.no_grad():
        if isinstance(module, torch.nn.Linear):
            module.weight.copy_(personalized_weights)
        elif hasattr(module, 'weight') and module.weight.dim() == 2:
            module.weight.copy_(personalized_weights)
        else:
            # Non-linear layers, no action taken
            pass

def compute_source_stats(model, device, data_loader, dataset_name, target_layer_name=None):
    """
    Compute source distribution statistics (public dataset)
    target_layer_name: Specify the layer name to extract features (for adaptive layer selection)
    """
    model = model.to(device)
    model.eval()

    extracted_features = []

    def hook_fn(module, input, output):
        # For OT personalization of linear layers, we need to extract the input features of the layer
        # Because OT is performed in the feature space (columns of weights)
        if isinstance(module, torch.nn.Linear):
            features = input[0] if isinstance(input, tuple) else input
        else:
            features = output
            
        if features.dim() == 4:
            if features.shape[2:] == (1, 1):
                features = features.squeeze(-1).squeeze(-1)
            else:
                features = features.mean(dim=[2, 3])  # Global Average Pooling
        extracted_features.append(features.detach().cpu())

    # If a target layer is specified, use it; otherwise, use the default layer
    if target_layer_name:
        # Get the module according to the layer name
        parts = target_layer_name.split('.')
        module = model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        handle = module.register_forward_hook(hook_fn)
    else:
        # Use default layer (for backward compatibility)
        if dataset_name == 'fmnist':
            handle = model.classifier[2].register_forward_hook(hook_fn)
        elif dataset_name == 'cifar10':
            handle = model.fc.register_forward_hook(hook_fn)
        elif dataset_name in ['tinyimagenet', 'imagenet']:
            handle = model.blocks[-1].register_forward_hook(hook_fn)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    all_features = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            extracted_features.clear()

            output = model(data)

            all_features.append(extracted_features[0])

    H = torch.cat(all_features, dim=0)  # shape: [N, D]
    N = H.size(0)
    mu_k = torch.mean(H, dim=0, keepdim=True)  # [1, D]
    centered = H - mu_k
    Sigma_k = (centered.T @ centered) / (N - 1)  # [D, D]

    handle.remove()

    return mu_k.squeeze(0), Sigma_k

def build_ot_source_distribution(weight_matrix):

    mass = torch.norm(weight_matrix, p=2, dim=1)
    mass = mass + 1e-6
    alpha = mass / mass.sum()
    # n = weight_matrix.size(0)
    # alpha = torch.full((n,), 1.0 / n, device=weight_matrix.device)
    return alpha, weight_matrix


def nearest_positive_definite_np(A):
    """
    Find the nearest positive-definite matrix to input A
    Use a more stable method to ensure the matrix is positive definite
    """
    B = (A + A.T) / 2
    
    # Add diagonal regularization to ensure numerical stability
    n = B.shape[0]
    eps = 1e-6
    B = B + eps * np.eye(n)
    
    try:
        # Attempt Cholesky decomposition to check if the matrix is positive definite
        np.linalg.cholesky(B)
        return B
    except np.linalg.LinAlgError:
        # If not positive definite, use eigenvalue decomposition
        try:
            eigvals, eigvecs = np.linalg.eigh(B)
            # Set negative and small eigenvalues to larger positive values
            eigvals[eigvals < 1e-2] = 1e-2
            A_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
            A_pd = (A_pd + A_pd.T) / 2  # Ensure symmetry
            # Again, add regularization
            A_pd = A_pd + eps * np.eye(n)
            return A_pd
        except:
            # If eigenvalue decomposition also fails, return a multiple of the identity matrix
            return np.eye(n) * 0.1

def build_ot_target_distribution(mu_k, Sigma_k, num_points):
    """
    Construct OT target distribution using a more stable sampling method
    """
    mu_np = mu_k.cpu().numpy()
    Sigma_np = Sigma_k.cpu().numpy()

    Sigma_pd = nearest_positive_definite_np(Sigma_np)
    
    # Use Cholesky decomposition for sampling, more stable
    try:
        L = np.linalg.cholesky(Sigma_pd)
        # Generate samples from standard normal distribution
        z = np.random.randn(num_points, len(mu_np))
        # Obtain target distribution samples through linear transformation
        samples = mu_np + z @ L.T
    except np.linalg.LinAlgError:
        # If Cholesky decomposition fails, use eigenvalue decomposition
        try:
            eigvals, eigvecs = np.linalg.eigh(Sigma_pd)
            eigvals = np.maximum(eigvals, 1e-6)  # Ensure positive eigenvalues
            L = eigvecs @ np.diag(np.sqrt(eigvals))
            z = np.random.randn(num_points, len(mu_np))
            samples = mu_np + z @ L.T
        except:
            # Final fallback: use identity covariance
            print("  WARNING: Using identity covariance for sampling")
            samples = mu_np + np.random.randn(num_points, len(mu_np)) * 0.1

    target_points = torch.tensor(samples, dtype=mu_k.dtype, device=mu_k.device)

    density = torch.norm(target_points, p=2, dim=1) + 1e-9
    beta = density / density.sum()

    return beta, target_points



def compute_ground_cost_matrix(source_points, target_points):

    target_points = target_points.to(source_points.device)
    n, d = source_points.shape
    m, _ = target_points.shape

    #  ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y^T
    x_sq = (source_points ** 2).sum(dim=1, keepdim=True)  # [n, 1]
    y_sq = (target_points ** 2).sum(dim=1, keepdim=True).T  # [1, m]
    cross = source_points @ target_points.T  # [n, m]

    M = x_sq + y_sq - 2 * cross  # [n, m]
    return M


def sinkhorn_ot(M, alpha, beta, epsilon, max_iter, tol):

    device = M.device
    alpha = alpha.to(device)
    beta = beta.to(device)

    # epsilon = M.mean().item() * 0.1  # or even 1.0

    K = torch.exp(-M / epsilon) + 1e-9

    u = torch.ones_like(alpha)
    v = torch.ones_like(beta)

    for _ in range(max_iter):
        u_prev = u.clone()
        v_prev = v.clone()

        u = alpha / (K @ v + 1e-9)
        v = beta / (K.T @ u + 1e-9)

        if torch.norm(u - u_prev) < tol and torch.norm(v - v_prev) < tol:
            break

    # T = diag(u) @ K @ diag(v)
    T = (u.unsqueeze(1) * K) * v.unsqueeze(0)

    return T



def replace_layer_weights_with_personalized(global_model, personalized_weights, dataset_name):

    if dataset_name == 'fmnist':
        target_layer = global_model.classifier[2]
    elif dataset_name in ['cifar10']:
        target_layer = global_model.fc
    elif dataset_name in ['tinyimagenet', 'imagenet']:
        target_layer = global_model.head  # ViT uses 'head' instead of 'fc'
    else:
        raise ValueError(f"Unsupported dataset for personalization: {dataset_name}")

    with torch.no_grad():
        target_layer.weight.copy_(personalized_weights)

    return global_model


def load_trainer_dataset(idx, dataset_name, non_iid_degree, data_dir='trainers_data'):
    data_path = os.path.join(data_dir, f'trainer_{idx}_data_{dataset_name}_{non_iid_degree}.pt')
    label_path = os.path.join(data_dir, f'trainer_{idx}_labels_{dataset_name}_{non_iid_degree}.pt')

    data = torch.load(data_path)
    labels = torch.load(label_path)

    dataset = TensorDataset(data, labels)
    return dataset

acc_list = []

if __name__ == '__main__':
    start = time.time()
    acc_log = []

    args = args_parser()
    # Dataset-model mapping: fmnist→LeNet, cifar10→ResNet18, tinyimagenet→ViT-B16
    if args.dataset in ['mnist', 'fmnist']:
        agent_model = model_2.LeNet5(args.num_classes)
    elif args.dataset in ['cifar10']:
        agent_model = model_2.Resnet18(args.num_classes)
    elif args.dataset in ['tinyimagenet', 'imagenet']:
        agent_model = model_2.ViT_B16()
    elif args.dataset in ['cifar100']:
        agent_model = model_2.VGG11(args.num_classes)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    dataset_train, dataset_test = dataset.get_dataset()

    public_dataset = subset_by_class_fraction(dataset_train, fraction_per_class=0.1)
    public_loader = torch.utils.data.DataLoader(public_dataset, batch_size=64, shuffle=True)
    S = sever.Sever(args, dataset_test, agent_model.state_dict())
    total_acc = S.evaluate()
    print(f"Initial Accuracy: {total_acc:.4f}")
    data_distributer = getattr(dataset_set, args.dataset)()
    data_num_dict = data_distributer.save_trainer_data()
    train_loader = data_distributer.get_tr_loaders()
    eval_loader = data_distributer.get_te_loader()

    trainer_dict = {idx: Trainer(args, agent_model, dataset_train, dataset_test, idx) for idx in range(args.trainer_num)}

    # for trainer_idx in trainer_dict.keys():
    #     trainer = trainer_dict[trainer_idx]
    #     trainer.save_class_data('trainers_data', trainer_idx)
    
    # Adaptive layer selection (round 1)
    selected_target_layers = None
    # Keep per-round layer stats for multi-layer OT personalization
    global_layer_mu_dict = None
    global_layer_Sigma_dict = None
    
    for round in range(args.round):
        print("---------------Round: {}------------------".format(round))
        mu_k_dict = dict()
        acc_dict = dict()
        Sigma_k_dict = dict()
        params_dict = dict()
        start_train = time.time()
        global_log_posterior = 0

        all_features = []
        all_labels = []
        all_softmax = []
        
        # Round 1: evaluate candidates and select layers
        if round == 0:
            print("Computing layer distinguishability for adaptive layer selection...")
            candidate_layers = get_candidate_layers(args.dataset)
            layer_mu_dict = {layer_name: {} for layer_name in candidate_layers}
            layer_Sigma_dict = {layer_name: {} for layer_name in candidate_layers}
            
            # Cache per-trainer features
            trainer_layer_features = {}
            trainer_labels = {}
            trainer_softmax = {}
            
            # Compute stats for all candidate layers per client
            for trainer_idx in trainer_dict.keys():
                trainer = trainer_dict[trainer_idx]
                
                # Extract multi-layer features
                acc, layer_mu_k, layer_Sigma_k, softmax_outputs, layer_features, labels, w = trainer.compute_feature_stats(
                    device=device,
                    data_loader=train_loader[trainer_idx],
                    dataset_name=args.dataset,
                    return_softmax=True,
                    target_layers=candidate_layers
                )
                
                # Store per-layer stats
                for layer_name in candidate_layers:
                    layer_mu_dict[layer_name][trainer_idx] = layer_mu_k[layer_name]
                    layer_Sigma_dict[layer_name][trainer_idx] = layer_Sigma_k[layer_name]
                
                # Save params and accuracy
                params_dict[trainer_idx] = w
                acc_dict[trainer_idx] = acc
                
                # Cache features, labels, softmax
                trainer_layer_features[trainer_idx] = layer_features
                trainer_labels[trainer_idx] = labels
                trainer_softmax[trainer_idx] = softmax_outputs
            
            # Compute S(l)
            S_dict = compute_layer_distinguishability(layer_mu_dict, layer_Sigma_dict, list(trainer_dict.keys()))
            
            # Select adaptive layers by coverage tau
            selected_target_layers = select_adaptive_layers(S_dict, tau=args.layer_coverage_tau)
            
            coverage_ratio = sum(S_dict[l] for l in selected_target_layers) / sum(S_dict.values())
            print(f"Layer distinguishability S(l): {S_dict}")
            print(f"Selected {len(selected_target_layers)} adaptive layer(s) with K*={len(selected_target_layers)}: {selected_target_layers}")
            print(f"Coverage ratio: {coverage_ratio:.2%} (threshold tau={args.layer_coverage_tau})")
            
            # Save global layer stats
            global_layer_mu_dict = layer_mu_dict
            global_layer_Sigma_dict = layer_Sigma_dict
            
            # Build final features from selected layers
            for trainer_idx in trainer_dict.keys():
                layer_features = trainer_layer_features[trainer_idx]
                
                if len(selected_target_layers) == 1:
                    # Single selected layer
                    selected_layer = selected_target_layers[0]
                    mu_k_dict[trainer_idx] = layer_mu_dict[selected_layer][trainer_idx]
                    Sigma_k_dict[trainer_idx] = layer_Sigma_dict[selected_layer][trainer_idx]
                    features = layer_features[selected_layer]
                else:
                    # Merge features for multiple layers
                    features_list = [layer_features[layer_name] for layer_name in selected_target_layers]
                    features = torch.cat(features_list, dim=1)
                    
                    # Recompute combined stats
                    N = features.size(0)
                    mu_k = torch.mean(features, dim=0, keepdim=True).squeeze(0)
                    centered = features - mu_k.unsqueeze(0)
                    Sigma_k = (centered.T @ centered) / (N - 1)
                    
                    mu_k_dict[trainer_idx] = mu_k
                    Sigma_k_dict[trainer_idx] = Sigma_k
                
                all_features.append(features)
                all_labels.append(trainer_labels[trainer_idx])
                all_softmax.append(trainer_softmax[trainer_idx])
            
        else:
            # Later rounds: use selected layers
            # Reset layer stats dicts
            layer_mu_dict_current = {layer_name: {} for layer_name in selected_target_layers}
            layer_Sigma_dict_current = {layer_name: {} for layer_name in selected_target_layers}
            
            for trainer_idx in trainer_dict.keys():
                trainer = trainer_dict[trainer_idx]

                # Always use selected layers
                acc, layer_mu_k, layer_Sigma_k, softmax_outputs, layer_features, labels, w = trainer.compute_feature_stats(
                    device=device,
                    data_loader=train_loader[trainer_idx],
                    dataset_name=args.dataset,
                    return_softmax=True,
                    target_layers=selected_target_layers
                )
                
                # Store per-layer stats
                for layer_name in selected_target_layers:
                    layer_mu_dict_current[layer_name][trainer_idx] = layer_mu_k[layer_name]
                    layer_Sigma_dict_current[layer_name][trainer_idx] = layer_Sigma_k[layer_name]
                
                if len(selected_target_layers) == 1:
                    # Single layer
                    selected_layer = selected_target_layers[0]
                    mu_k = layer_mu_k[selected_layer]
                    Sigma_k = layer_Sigma_k[selected_layer]
                    features = layer_features[selected_layer]
                else:
                    # Multi-layer: concat features
                    features_list = [layer_features[layer_name] for layer_name in selected_target_layers]
                    features = torch.cat(features_list, dim=1)
                    
                    # Recompute stats
                    N = features.size(0)
                    mu_k = torch.mean(features, dim=0, keepdim=True).squeeze(0)
                    centered = features - mu_k.unsqueeze(0)
                    Sigma_k = (centered.T @ centered) / (N - 1)
                
                mu_k_dict[trainer_idx] = mu_k
                Sigma_k_dict[trainer_idx] = Sigma_k
                acc_dict[trainer_idx] = acc
                params_dict[trainer_idx] = w
                
                all_features.append(features)
                all_labels.append(labels)
                all_softmax.append(softmax_outputs)

                total_correct = 0
                total_samples = 0
                if round == args.round - 1:
                    train_loaders = data_distributer.get_tr_loaders()

                    for other_idx in range(len(train_loaders)):
                        if other_idx == trainer_idx:
                            continue

                        other_dataset = train_loaders[other_idx].dataset
                        # print(len(other_dataset))
                        S = sever.Sever(args, other_dataset, w)
                        correct, total = S.evaluate(return_raw=True)

                        total_correct += correct
                        total_samples += total

                    remain_acc = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
                    print("----Round: {}----  trainer: {} ----Train Acc: {:.2f}----- Remain Acc: {:.2f}------".format(round,
                                                                                                                      trainer_idx,
                                                                                                                      acc,
                                                                                                                      remain_acc))

            # Update global layer stats
            global_layer_mu_dict = layer_mu_dict_current
            global_layer_Sigma_dict = layer_Sigma_dict_current


            mu_k_dict[trainer_idx] = mu_k
            Sigma_k_dict[trainer_idx] = Sigma_k
            acc_dict[trainer_idx] = acc
            params_dict[trainer_idx] = w

            all_features.append(features)
            all_labels.append(labels)
            all_softmax.append(softmax_outputs)

        features_all = torch.cat(all_features, dim=0)
        labels_all = torch.cat(all_labels, dim=0)
        softmax_all = torch.cat(all_softmax, dim=0)

        pseudo_dataset = PseudoFeatureDataset(
            features=features_all,
            labels=labels_all,
            logits=softmax_all,  # soft labels
            use_soft_labels=True  # Use soft labels
        )
        pseudo_loader = DataLoader(pseudo_dataset, batch_size=64, shuffle=True)

        params = list(params_dict.values())
        global_model = aggregate_model(params, data_num_dict)

        new_state_dict, acc = distill_train_and_evaluate(
            state_dict=global_model,
            device=device,
            public_loader=public_loader,
            pseudo_dataset=pseudo_dataset,
            dataset_name=args.dataset,
            alpha=1,  # soft label loss
            beta=0.5,  # hard label loss
            gamma=1,  # public CE loss
            temperature=1.0,
            steps=1
        )

        acc_list.append(acc)

        print(f'----total acc:{acc:.2f}----')
        round_acc = {'round': round}
        for k in acc_dict:
            round_acc[f'client_{k}_acc'] = acc_dict[k]
        round_acc['global_acc'] = acc
        acc_log.append(round_acc)

        agent_model.load_state_dict(new_state_dict)

        # for trainer_idx in trainer_dict.keys():
        #     trainer = trainer_dict[trainer_idx]
        #     trainer.set_model_params(agent_model.state_dict())
        W = extract_fc_layer_weights_by_dataset(agent_model, args.dataset)

        # OT personalization for each selected layer
        for trainer_idx in trainer_dict.keys():
            personalized_model = copy.deepcopy(agent_model)
            
            # Per-layer OT personalization
            for selected_layer in selected_target_layers:
                # Extract layer weights
                W = extract_layer_weights(personalized_model, args.dataset, selected_layer)
                
                # Skip non-linear layers (e.g., conv)
                if W is None:
                    continue
                
                # Source stats on public data
                mu_k_pub, Sigma_k_pub = compute_source_stats(personalized_model, device, public_loader, args.dataset, target_layer_name=selected_layer)
                
                # OT on weight row space
                num_samples = W.size(0)  # number of output units (rows)
                alpha, source_points = build_ot_target_distribution(mu_k_pub, Sigma_k_pub, num_samples)
                
                # Client layer stats
                mu_k_client = global_layer_mu_dict[selected_layer][trainer_idx]
                Sigma_k_client = global_layer_Sigma_dict[selected_layer][trainer_idx]
                
                beta, target_points = build_ot_target_distribution(mu_k_client, Sigma_k_client, num_samples)
                
                # Ground cost matrix
                M = compute_ground_cost_matrix(source_points, target_points)
                
                # Sinkhorn OT
                T = sinkhorn_ot(M, alpha, beta, epsilon=0.01, max_iter=100, tol=1e-6)
                T = T.to(device)
                W = W.to(device)
                
                # Personalized weights
                personalized_weights = (T * T.size(0)) @ W
                
                # Replace layer weights
                replace_single_layer_weights(personalized_model, personalized_weights, args.dataset, selected_layer)

            # Update client model
            trainer = trainer_dict[trainer_idx]
            trainer.set_model_params(personalized_model.state_dict())

            # S = sever.Sever(args, dataset_test, personalized_model.state_dict())
            # global_acc = S.evaluate()
            # print("----Round: {}----  trainer: {}  Global Acc: {:.2f}------".format(round, trainer_idx, global_acc))

            # G_personalized = replace_layer_weights_with_personalized(agent_model, personalized_weights, args.dataset)

            # trainer = trainer_dict[trainer_idx]
            # trainer.set_model_params(G_personalized.state_dict())
        # print(personalized_weights)
        # agent_model.load_state_dict(new_state_dict)

    save_dir = './Intermediate_data'
    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(acc_log)
    excel_save_path = os.path.join(save_dir, f'accuracy_{args.dataset}_{args.non_iid_degree}.xlsx')
    df.to_excel(excel_save_path, index=False)

    end = time.time()
    print("Total time cost: {:.2f}".format(end - start))
    plt.figure()
    plt.plot(acc_list, label='Fedkd')
    plt.title('Global Model Accuracy')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(False)
    plt.show()