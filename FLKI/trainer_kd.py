import copy
import random
import os
import math
import torch
import torch.utils.data as data
from args import args
import torch.nn as nn


class Trainer(object):
    def __init__(self, args, model, train_data, eval_data, id=-1):
        self.trainer_id = id
        self.train_data = train_data
        self.eval_data = eval_data
        self.local_model = copy.deepcopy(model)
        # Default federated config (used by tinyimagenet local_update)
        self.fed_config = self._default_fed_config(args)


    def _get_indices_by_class(self, data, trainer_id):
        """Get data indices for a specific trainer's label."""
        targets = data.targets
        indices = [i for i, label in enumerate(targets) if label == trainer_id]
        return indices

    def set_model_params(self, params):
        """Update local model parameters."""
        self.local_model.load_state_dict(copy.deepcopy(params))

    def save_class_data(self, save_dir, idx):
        """Save training data to disk."""
        os.makedirs(save_dir, exist_ok=True)
        data_list = []
        labels_list = []
        for data, labels in self.train_loader:
            data_list.append(data)
            labels_list.append(labels)

        data_tensor = torch.cat(data_list)
        labels_tensor = torch.cat(labels_list)

        torch.save(data_tensor, os.path.join(save_dir, f'class_{idx}_data_{args.dataset}_{args.non_iid_degree}.pt'))
        torch.save(labels_tensor, os.path.join(save_dir, f'class_{idx}_labels_{args.dataset}_{args.non_iid_degree}.pt'))

        print(f"Trainer {idx} total data: {len(data_tensor)}")

    def compute_feature_stats(self, device, data_loader, dataset_name, return_softmax=False, target_layers=None):
        """
        Compute feature statistics for knowledge distillation.
        
        Args:
            target_layers: List of layer names for multi-layer extraction
        """
        # For tinyimagenet, first perform local training via local_update,
        # then run forward-only to collect statistics.
        if dataset_name == 'tinyimagenet':
            # Train locally using the specialized routine
            _state_dict, _acc, _avg_loss = self.local_update(device, data_loader)
            model = self.local_model.to(device)
            model.eval()

            extracted_features = []
            all_logits = []
            all_labels = []
            all_features = []

            def hook_fn(module, input, output):
                features = input[0] if isinstance(input, tuple) else input
                if features.dim() == 4:
                    if features.shape[2:] == (1, 1):
                        features = features.squeeze(-1).squeeze(-1)
                    else:
                        features = features.mean(dim=[2, 3])
                extracted_features.append(features.detach().cpu())

            handle = model.blocks[-1].register_forward_hook(hook_fn)

            correct = 0
            total_samples = 0

            with torch.no_grad():
                for data, target in data_loader:
                    data, target = data.to(device), target.to(device)
                    if data.size(0) == 1:
                        continue
                    extracted_features.clear()
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total_samples += data.size(0)
                    all_features.append(extracted_features[0])
                    all_labels.append(target.cpu())
                    all_logits.append(torch.softmax(output, dim=1).cpu() if return_softmax else output.cpu())

            features = torch.cat(all_features, dim=0)
            logits_or_softmax = torch.cat(all_logits, dim=0)
            labels = torch.cat(all_labels, dim=0)

            N = features.size(0)
            mu_k = torch.mean(features, dim=0, keepdim=True)
            centered = features - mu_k
            Sigma_k = (centered.T @ centered) / (N - 1)

            handle.remove()

            accuracy = 100.0 * correct / total_samples if total_samples > 0 else 0.0
            return accuracy, mu_k.squeeze(0), Sigma_k, logits_or_softmax, features, labels, model.state_dict()

        # Non-tinyimagenet path: original behavior
        model = self.local_model.to(device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        Loss = torch.nn.CrossEntropyLoss()
        
        # Multi-layer extraction if target_layers specified
        if target_layers is not None:
            return self._compute_multi_layer_stats(model, optimizer, Loss, device, data_loader, 
                                                   dataset_name, target_layers, return_softmax)
        
        extracted_features = []
        all_logits = []
        all_labels = []
        all_features = []

        # Register hook for target layer extraction
        def hook_fn(module, input, output):
            if dataset_name in ['cifar10', 'tinyimagenet', 'imagenet']:
                features = input[0] if isinstance(input, tuple) else input
            elif dataset_name == 'fmnist' and hasattr(module, 'in_features'):
                features = input[0] if isinstance(input, tuple) else input
            else:
                features = output
            
            if features.dim() == 4 and features.shape[2:] == (1, 1):
                features = features.squeeze(-1).squeeze(-1)
            extracted_features.append(features.detach().cpu())

        # Register hook for model-specific target layer
        if dataset_name == 'fmnist':
            handle = model.classifier[2].register_forward_hook(hook_fn)
        elif dataset_name == 'cifar10':
            handle = model.fc.register_forward_hook(hook_fn)
        elif dataset_name in ['tinyimagenet', 'imagenet']:
            handle = model.blocks[-1].register_forward_hook(hook_fn)

        correct = 0
        total_samples = 0

        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            # Skip batch size 1 to avoid BatchNorm errors
            if data.size(0) == 1:
                continue
            
            extracted_features.clear()

            optimizer.zero_grad()
            output = model(data)
            loss = Loss(output, target)
            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size(0)

            all_features.append(extracted_features[0])
            all_labels.append(target.cpu())
            if return_softmax:
                all_logits.append(torch.softmax(output, dim=1).cpu())
            else:
                all_logits.append(output.cpu())

        # (keep original training-based collection; avoid duplicate re-append of logits)

        # Concatenate results
        features = torch.cat(all_features, dim=0)
        logits_or_softmax = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)

        N = features.size(0)
        mu_k = torch.mean(features, dim=0, keepdim=True)
        centered = features - mu_k
        Sigma_k = (centered.T @ centered) / (N - 1)

        handle.remove()

        accuracy = 100.0 * correct / total_samples

        return accuracy, mu_k.squeeze(0), Sigma_k, logits_or_softmax, features, labels, model.state_dict()

    def _compute_multi_layer_stats(self, model, optimizer, Loss, device, data_loader, 
                                   dataset_name, target_layers, return_softmax=False):
        """Compute multi-layer feature statistics."""
        # For tinyimagenet, train via local_update first, then eval-only pass.
        if dataset_name == 'tinyimagenet':
            _state_dict, _acc, _avg_loss = self.local_update(device, data_loader)
            model = self.local_model.to(device)
            model.eval()
            layer_features = {layer_name: [] for layer_name in target_layers}
            all_logits = []
            all_labels = []
            handles = []

            def make_hook(layer_name):
                def hook_fn(module, input, output):
                    if isinstance(module, torch.nn.Linear):
                        features = input[0] if isinstance(input, tuple) else input
                    else:
                        features = output
                    if features.dim() == 4:
                        if features.shape[2:] == (1, 1):
                            features = features.squeeze(-1).squeeze(-1)
                        else:
                            features = features.mean(dim=[2, 3])
                    layer_features[layer_name].append(features.detach().cpu())
                return hook_fn

            for layer_name in target_layers:
                layer_module = self._get_layer_by_name(model, layer_name)
                handle = layer_module.register_forward_hook(make_hook(layer_name))
                handles.append(handle)

            correct = 0
            total_samples = 0
            with torch.no_grad():
                for data, target in data_loader:
                    data, target = data.to(device), target.to(device)
                    if data.size(0) == 1:
                        continue
                    for layer_name in target_layers:
                        if layer_features[layer_name] and len(layer_features[layer_name]) > 0:
                            layer_features[layer_name] = layer_features[layer_name][-1:]
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total_samples += data.size(0)
                    all_labels.append(target.cpu())
                    all_logits.append(torch.softmax(output, dim=1).cpu() if return_softmax else output.cpu())

            layer_mu_k = {}
            layer_Sigma_k = {}
            layer_all_features = {}
            for layer_name in target_layers:
                features = torch.cat(layer_features[layer_name], dim=0)
                if features.dim() > 2:
                    N = features.size(0)
                    features = features.view(N, -1)
                N = features.size(0)
                mu_k = torch.mean(features, dim=0, keepdim=True)
                centered = features - mu_k
                Sigma_k = (centered.T @ centered) / (N - 1)
                layer_mu_k[layer_name] = mu_k.squeeze(0)
                layer_Sigma_k[layer_name] = Sigma_k
                layer_all_features[layer_name] = features

            logits_or_softmax = torch.cat(all_logits, dim=0)
            labels = torch.cat(all_labels, dim=0)
            for handle in handles:
                handle.remove()
            accuracy = 100.0 * correct / total_samples if total_samples > 0 else 0.0
            return accuracy, layer_mu_k, layer_Sigma_k, logits_or_softmax, layer_all_features, labels, model.state_dict()

        # Non-tinyimagenet path: original behavior
        layer_features = {layer_name: [] for layer_name in target_layers}
        all_logits = []
        all_labels = []
        handles = []
        
        # Register hooks for each target layer
        def make_hook(layer_name):
            def hook_fn(module, input, output):
                # Extract input features for linear layers, output for others
                if isinstance(module, torch.nn.Linear):
                    features = input[0] if isinstance(input, tuple) else input
                else:
                    features = output
                
                if features.dim() == 4:
                    if features.shape[2:] == (1, 1):
                        features = features.squeeze(-1).squeeze(-1)
                    else:
                        # Global average pooling for larger feature maps
                        features = features.mean(dim=[2, 3])
                
                layer_features[layer_name].append(features.detach().cpu())
            return hook_fn
        
        # Register hooks for all target layers
        for layer_name in target_layers:
            layer_module = self._get_layer_by_name(model, layer_name)
            handle = layer_module.register_forward_hook(make_hook(layer_name))
            handles.append(handle)
        
        correct = 0
        total_samples = 0
        
        # Training and feature extraction
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            # Skip batch size 1 to avoid BatchNorm errors
            if data.size(0) == 1:
                continue
            
            # Clear previous batch features
            for layer_name in target_layers:
                if layer_features[layer_name] and len(layer_features[layer_name]) > 0:
                    layer_features[layer_name] = layer_features[layer_name][-1:]
            
            optimizer.zero_grad()
            output = model(data)
            loss = Loss(output, target)
            loss.backward()
            optimizer.step()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size(0)
            
            all_labels.append(target.cpu())
            if return_softmax:
                all_logits.append(torch.softmax(output, dim=1).cpu())
            else:
                all_logits.append(output.cpu())
        
        # Compute statistics for each layer
        layer_mu_k = {}
        layer_Sigma_k = {}
        layer_all_features = {}
        
        for layer_name in target_layers:
            features = torch.cat(layer_features[layer_name], dim=0)
            
            # Flatten if multi-dimensional
            if features.dim() > 2:
                N = features.size(0)
                features = features.view(N, -1)
            
            N = features.size(0)
            mu_k = torch.mean(features, dim=0, keepdim=True)
            centered = features - mu_k
            Sigma_k = (centered.T @ centered) / (N - 1)
            
            layer_mu_k[layer_name] = mu_k.squeeze(0)
            layer_Sigma_k[layer_name] = Sigma_k
            layer_all_features[layer_name] = features
        
        logits_or_softmax = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        # Remove all hooks
        for handle in handles:
            handle.remove()
        
        accuracy = 100.0 * correct / total_samples
        
        return accuracy, layer_mu_k, layer_Sigma_k, logits_or_softmax, layer_all_features, labels, model.state_dict()

    def _default_fed_config(self, args):
        """Provide default federated config for local_update."""
        return {
            'optimizer': 'SGD',
            'local_lr': args.lr,
            'weight_decay': 0.0,
            'betas': (0.9, 0.999),
            'finetune_strategy': 'none',
            'warmup_epochs': 0,
            'local_epochs': 1,
            'min_lr': 0.0,
            'label_smoothing': 0.0,
            'gradient_clip': 0.0,
            'freeze_backbone': False,
        }
    
    def local_update(self, device, train_loader):
        """Local federated training update method."""
        config = self.fed_config
    
        # Get model
        model = self.local_model
        model.to(device)
        model.train()
    
        # Setup partial finetune if enabled
        if config['finetune_strategy'] == 'partial':
            self._setup_partial_finetune(model)
    
        # ========== Parameter groups ==========
        param_groups = self._get_param_groups(model)
    
        # If no groups, use all params
        if not param_groups:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            if trainable_params:
                param_groups = [{
                    'params': trainable_params,
                    'lr': config['local_lr'],
                    'weight_decay': config['weight_decay']
                }]
            else:
                # If no trainable params, use all parameters
                param_groups = [{
                    'params': list(model.parameters()),
                    'lr': config['local_lr'],
                    'weight_decay': config['weight_decay']
                }]
    
        # ========== Optimizer setup ==========
        if config['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=config['local_lr'],
                weight_decay=config['weight_decay'],
                betas=config['betas']
            )
        else:
            optimizer = torch.optim.SGD(
                param_groups,
                lr=config['local_lr'],
                momentum=0.9,
                weight_decay=config['weight_decay']
            )
    
        # ========== Learning rate schedule ==========
        total_steps = config['local_epochs'] * len(train_loader)
        warmup_steps = config['warmup_epochs'] * len(train_loader)
    
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            # Cosine decay
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(config['min_lr'] / config['local_lr'],
                       0.5 * (1.0 + math.cos(math.pi * progress)))
    
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
        # ========== Loss ==========
        criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    
        # ========== Training stats ==========
        train_loss = 0.0
        correct = 0
        data_size = 0
    
        # ========== Training loop ==========
        for epoch in range(config['local_epochs']):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
    
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
    
                optimizer.zero_grad()
                output = model(data)  # Use local model
                loss = criterion(output, target)
                loss.backward()
    
                # Gradient clipping
                if config['gradient_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
    
                optimizer.step()
                scheduler.step()
    
                # Stats
                epoch_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                epoch_correct += pred.eq(target.view_as(pred)).sum().item()
                epoch_total += target.size(0)
    
    
            # Epoch log
            epoch_acc = 100. * epoch_correct / epoch_total
            current_lr = optimizer.param_groups[0]['lr']
            print(f'    Client {self.trainer_id}, Epoch {epoch + 1}: '
                  f'Loss={epoch_loss / len(train_loader):.4f}, '
                  f'Acc={epoch_acc:.2f}%, LR={current_lr:.2e}')
    
            train_loss += epoch_loss
            correct += epoch_correct
            data_size += epoch_total
    
        # Averages
        avg_loss = train_loss / (config['local_epochs'] * len(train_loader))
        accuracy = 100.0 * correct / data_size
    
        return model.state_dict(), accuracy, avg_loss
    
    def _setup_partial_finetune(self, model):
        """Setup partial finetune."""
        config = self.fed_config
    
        # Unfreeze all params first
        for param in model.parameters():
            param.requires_grad = True
    
        if config.get('freeze_backbone', False):
            # Freeze all blocks except the last two
            if hasattr(model, 'blocks'):
                for i, block in enumerate(model.blocks):
                    if i < len(model.blocks) - 2:
                        for param in block.parameters():
                            param.requires_grad = False
    
            # Freeze patch embedding
            if hasattr(model, 'patch_embed'):
                for param in model.patch_embed.parameters():
                    param.requires_grad = False
    
        # Classification head is always trainable
        if hasattr(model, 'head'):
            for param in model.head.parameters():
                param.requires_grad = True
    
    
    def _get_param_groups(self, model):
        """Get parameter groups."""
        config = self.fed_config
        param_groups = []
        used_params = set()
    
        # 1. Head params
        if hasattr(model, 'head'):
            head_params = []
            for param in model.head.parameters():
                if param.requires_grad and id(param) not in used_params:
                    head_params.append(param)
                    used_params.add(id(param))
    
            if head_params:
                param_groups.append({
                    'params': head_params,
                    'lr': config['local_lr'] * 10,  # higher LR for head
                    'weight_decay': config['weight_decay']
                })
    
        # 2. Last two Transformer blocks
        if hasattr(model, 'blocks') and len(model.blocks) >= 2:
            block_params = []
            for block in model.blocks[-2:]:  # last two blocks
                for param in block.parameters():
                    if param.requires_grad and id(param) not in used_params:
                        block_params.append(param)
                        used_params.add(id(param))
    
            if block_params:
                param_groups.append({
                    'params': block_params,
                    'lr': config['local_lr'],  # medium LR
                    'weight_decay': config['weight_decay']
                })
    
        # 3. Other trainable params (lower LR)
        other_params = []
        for param in model.parameters():
            if param.requires_grad and id(param) not in used_params:
                other_params.append(param)
    
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': config['local_lr'] * 0.5,  # lower LR
                'weight_decay': config['weight_decay']
            })
    
        # 4. Fallback: use all params
        if not param_groups:
            all_params = list(model.parameters())
            if all_params:
                param_groups.append({
                    'params': all_params,
                    'lr': config['local_lr'],
                    'weight_decay': config['weight_decay']
                })
    
        return param_groups

    def _get_layer_by_name(self, model, layer_name):
        """Get layer module by name."""
        parts = layer_name.split('.')
        module = model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def clip_gradients(self, model):
        """Clip gradients for differential privacy."""
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.dp_clip)









