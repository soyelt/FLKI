import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, BatchNorm2d, ReLU, AdaptiveAvgPool2d
from einops import rearrange

class MNIST_CNN_Net(nn.Module):
    def __init__(self):
        super(MNIST_CNN_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(5 * 5 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5 * 5 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class LeNet5(nn.Module):
    def __init__(self, num_classes):
        """
        num_classes: 分类的数量
        grayscale：是否为灰度图
        """
        super(LeNet5, self).__init__()

        self.num_classes = num_classes

        # 卷积神经网络，添加了 ReLU 激活函数
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(inplace=True),  # 添加 ReLU 激活
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),  # 添加 ReLU 激活
            nn.MaxPool2d(kernel_size=2)  # 原始的模型使用的是 平均池化
        )

        # 分类器，添加了 ReLU 激活函数
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),  # 添加 ReLU 激活
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),  # 添加 ReLU 激活
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)  # 输出 16*5*5 特征图
        x = torch.flatten(x, 1)  # 展平 （1， 16*5*5）
        logits = self.classifier(x)  # 输出 num_classes
        return logits


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        x = rearrange(x, 'b c h w -> b (h w) c')  # 展平成 (B, n_patches, embed_dim)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_dim=3072, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViT_B16(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=200,
                 embed_dim=768, depth=12, num_heads=12, mlp_dim=3072, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        # ========== 联邦学习固定配置 ==========
        self.fed_config = {
            # 本地训练配置
            'local_epochs': 1,  # 每个客户端训练3轮
            'local_lr': 1e-4,  # 本地学习率
            'local_batch_size': 8,  # 客户端本地batch

            # 优化器配置
            'optimizer': 'AdamW',  # ViT必须使用AdamW
            'weight_decay': 0.05,  # 权重衰减
            'betas': (0.9, 0.999),  # AdamW参数

            # 学习率调度
            'warmup_epochs': 1,  # 预热1个epoch
            'warmup_lr': 1e-6,  # 预热学习率
            'min_lr': 1e-6,  # 最小学习率

            # 梯度处理
            'gradient_clip': 1.0,  # 梯度裁剪

            # 正则化
            'label_smoothing': 0.1,  # 标签平滑

            # 模型微调策略
            'finetune_strategy': 'partial',  # 部分微调
            'freeze_backbone': False,  # 是否冻结backbone
            'trainable_layers': ['head', 'blocks.10', 'blocks.11'],  # 可训练层
        }

        # 设置可训练参数
        self.setup_fed_training()

    def setup_fed_training(self):
        """设置联邦学习训练参数"""
        config = self.fed_config

        if config['finetune_strategy'] == 'partial':
            # 冻结部分层，只训练分类头和最后几层
            self._freeze_layers()

        # 设置参数组（差分学习率）
        self.param_groups = self._get_param_groups()

    def _freeze_layers(self):
        """冻结不需要训练的参数"""
        config = self.fed_config

        # 冻结所有Transformer块
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False

        # 解冻最后两个Transformer块
        for param in self.blocks[-2:].parameters():
            param.requires_grad = True

        # 冻结patch embedding和position embedding
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        self.pos_embed.requires_grad = False
        self.cls_token.requires_grad = False

        # 分类头总是可训练
        for param in self.head.parameters():
            param.requires_grad = True

    def _get_param_groups(self):
        """获取不同学习率的参数组，确保参数不重复"""
        config = self.fed_config
        param_groups = []

        # 使用集合跟踪已分配的参数
        used_params = set()

        # 1. 分类头：最高学习率
        head_params = []
        for param in self.head.parameters():
            if id(param) not in used_params:
                head_params.append(param)
                used_params.add(id(param))

        if head_params:
            param_groups.append({
                'params': head_params,
                'lr': config['local_lr'] * 10,
                'weight_decay': config['weight_decay']
            })

        # 2. 最后两个Transformer块：中等学习率
        if len(self.blocks) >= 2:
            block_params = []
            for block in self.blocks[-2:]:
                for param in block.parameters():
                    if id(param) not in used_params:
                        block_params.append(param)
                        used_params.add(id(param))

            if block_params:
                param_groups.append({
                    'params': block_params,
                    'lr': config['local_lr'],
                    'weight_decay': config['weight_decay']
                })

        # 3. 最后两个块的norm层：单独设置
        norm_params = []

        if len(self.blocks) >= 1:
            # 最后一层的norm
            for param in self.blocks[-1].norm1.parameters():
                if id(param) not in used_params:
                    norm_params.append(param)
                    used_params.add(id(param))

            for param in self.blocks[-1].norm2.parameters():
                if id(param) not in used_params:
                    norm_params.append(param)
                    used_params.add(id(param))

        if len(self.blocks) >= 2:
            # 倒数第二层的norm
            for param in self.blocks[-2].norm1.parameters():
                if id(param) not in used_params:
                    norm_params.append(param)
                    used_params.add(id(param))

            for param in self.blocks[-2].norm2.parameters():
                if id(param) not in used_params:
                    norm_params.append(param)
                    used_params.add(id(param))

        if norm_params:
            param_groups.append({
                'params': norm_params,
                'lr': config['local_lr'] * 2,
                'weight_decay': 0.0
            })

        # 4. 其他参数（较低学习率）
        other_params = []
        for param in self.parameters():
            if param.requires_grad and id(param) not in used_params:
                other_params.append(param)

        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': config['local_lr'] * 0.5,
                'weight_decay': config['weight_decay']
            })

        # 5. 检查是否所有可训练参数都已分配
        all_trainable_params = [p for p in self.parameters() if p.requires_grad]
        assigned_params = sum([len(group['params']) for group in param_groups])

        if assigned_params < len(all_trainable_params):
            print(f"Warning: {len(all_trainable_params) - assigned_params} parameters not assigned to any group")

        return param_groups



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]
        return self.head(cls_out)

    def local_update(self, device, train_loader):
        """联邦学习的本地更新方法"""
        config = self.fed_config

        self.to(device)
        self.train()

        # 设置部分微调（如果启用）
        if config['finetune_strategy'] == 'partial':
            self._setup_partial_finetune()

        # ========== 获取参数组 ==========
        param_groups = self._get_param_groups()

        # 如果没有参数组，使用所有参数
        if not param_groups:
            trainable_params = [p for p in self.parameters() if p.requires_grad]
            if trainable_params:
                param_groups = [{
                    'params': trainable_params,
                    'lr': config['local_lr'],
                    'weight_decay': config['weight_decay']
                }]
            else:
                # 如果没有可训练参数，使用所有参数
                param_groups = [{
                    'params': list(self.parameters()),
                    'lr': config['local_lr'],
                    'weight_decay': config['weight_decay']
                }]

        # ========== 优化器设置 ==========
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

        # ========== 学习率调度 ==========
        total_steps = config['local_epochs'] * len(train_loader)
        warmup_steps = config['warmup_epochs'] * len(train_loader)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # 线性预热
                return float(current_step) / float(max(1, warmup_steps))
            # 余弦退火
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(config['min_lr'] / config['local_lr'],
                       0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # ========== 损失函数 ==========
        criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])

        # ========== 训练统计 ==========
        train_loss = 0.0
        correct = 0
        data_size = 0

        # ========== 训练循环 ==========
        for epoch in range(config['local_epochs']):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = self(data)
                loss = criterion(output, target)
                loss.backward()

                # 梯度裁剪
                if config['gradient_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), config['gradient_clip'])

                optimizer.step()
                scheduler.step()

                # 统计
                epoch_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                epoch_correct += pred.eq(target.view_as(pred)).sum().item()
                epoch_total += target.size(0)

            # 打印epoch信息
            epoch_acc = 100. * epoch_correct / epoch_total
            current_lr = optimizer.param_groups[0]['lr']
            print(f'    Epoch {epoch + 1}: Loss={epoch_loss / len(train_loader):.4f}, '
                  f'Acc={epoch_acc:.2f}%, LR={current_lr:.2e}')

            train_loss += epoch_loss
            correct += epoch_correct
            data_size += epoch_total

        # 计算平均值
        avg_loss = train_loss / (config['local_epochs'] * len(train_loader))
        accuracy = 100.0 * correct / data_size

        return self.state_dict(), accuracy, avg_loss

    def _setup_partial_finetune(self):
        """设置部分微调：冻结不需要训练的参数"""
        config = self.fed_config

        # 首先解冻所有参数
        for param in self.parameters():
            param.requires_grad = True

        # 根据配置冻结层
        if config.get('freeze_backbone', False):
            # 冻结所有Transformer块
            for block in self.blocks:
                for param in block.parameters():
                    param.requires_grad = False

            # 冻结patch embedding和position embedding
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            self.pos_embed.requires_grad = False
            self.cls_token.requires_grad = False

        # 根据trainable_layers设置可训练层
        trainable_layers = config.get('trainable_layers', ['head', 'blocks.10', 'blocks.11'])

        # 首先冻结所有
        for name, param in self.named_parameters():
            param.requires_grad = False

        # 解冻指定的层
        for layer_name in trainable_layers:
            if layer_name == 'head':
                for param in self.head.parameters():
                    param.requires_grad = True
            elif layer_name.startswith('blocks.'):
                try:
                    layer_idx = int(layer_name.split('.')[-1])
                    if 0 <= layer_idx < len(self.blocks):
                        for param in self.blocks[layer_idx].parameters():
                            param.requires_grad = True
                except (ValueError, IndexError):
                    print(f"Warning: Invalid layer index in trainable_layers: {layer_name}")




class CIFAR_CNN_Net(torch.nn.Module):
    def __init__(self):
        super(CIFAR_CNN_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.fc1 = nn.Linear(1600, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 1600)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # return x
        return F.log_softmax(x, dim=1)

class VGG11(nn.Module):
    def __init__(self, num_classes):
        super(VGG11, self).__init__()
        self.conv_layer1 = self._make_conv_1(3,64)
        self.conv_layer2 = self._make_conv_1(64,128)
        self.conv_layer3 = self._make_conv_2(128,256)
        self.conv_layer4 = self._make_conv_2(256,512)
        self.conv_layer5 = self._make_conv_2(512,512)
        self.classifier = nn.Sequential(
            nn.Linear(512, 64),    # 这里修改一下输入输出维度
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes)
        )

    def _make_conv_1(self,in_channels,out_channels):
        layer = nn.Sequential(
                nn.Conv2d(in_channels,out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        return layer
    def _make_conv_2(self,in_channels,out_channels):
        layer = nn.Sequential(
                nn.Conv2d(in_channels,out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels,out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
              )
        return layer

    def forward(self, x):
        # 32*32 channel == 3
        x = self.conv_layer1(x)
        # 16*16 channel == 64
        x = self.conv_layer2(x)
        # 8*8 channel == 128
        x = self.conv_layer3(x)
        # 4*4 channel == 256
        x = self.conv_layer4(x)
        # 2*2 channel == 512
        x = self.conv_layer5(x)
        # 1*1 channel == 512
        x = x.view(x.size(0), -1)
        # 512
        x = self.classifier(x)
        # 10
        return x

class Resnet18(nn.Module):
    def __init__(self, num_classes):
        super(Resnet18, self).__init__()
        self.model0 = Sequential(
            # 0
            # 输入3通道、输出64通道、卷积核大小、步长、补零、
            Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=2, padding=3),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
        )
        self.model1 = Sequential(
            # 1.1
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R1 = ReLU()

        self.model2 = Sequential(
            # 1.2
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R2 = ReLU()

        self.model3 = Sequential(
            # 2.1
            Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.en1 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R3 = ReLU()

        self.model4 = Sequential(
            # 2.2
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R4 = ReLU()

        self.model5 = Sequential(
            # 3.1
            Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.en2 = Sequential(
            Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R5 = ReLU()

        self.model6 = Sequential(
            # 3.2
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R6 = ReLU()

        self.model7 = Sequential(
            # 4.1
            Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.en3 = Sequential(
            Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R7 = ReLU()

        self.model8 = Sequential(
            # 4.2
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R8 = ReLU()

        # AAP 自适应平均池化
        self.aap = AdaptiveAvgPool2d((1, 1))
        # flatten 维度展平
        self.flatten = Flatten(start_dim=1)
        # FC 全连接层
        self.fc = Linear(512, num_classes)

    def forward(self, x):
        x = self.model0(x)

        f1 = x
        x = self.model1(x)
        x = x + f1
        x = self.R1(x)

        f1_1 = x
        x = self.model2(x)
        x = x + f1_1
        x = self.R2(x)

        f2_1 = x
        f2_1 = self.en1(f2_1)
        x = self.model3(x)
        x = x + f2_1
        x = self.R3(x)

        f2_2 = x
        x = self.model4(x)
        x = x + f2_2
        x = self.R4(x)

        f3_1 = x
        f3_1 = self.en2(f3_1)
        x = self.model5(x)
        x = x + f3_1
        x = self.R5(x)

        f3_2 = x
        x = self.model6(x)
        x = x + f3_2
        x = self.R6(x)

        f4_1 = x
        f4_1 = self.en3(f4_1)
        x = self.model7(x)
        x = x + f4_1
        x = self.R7(x)

        f4_2 = x
        x = self.model8(x)
        x = x + f4_2
        x = self.R8(x)

        # 最后3个
        x = self.aap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
