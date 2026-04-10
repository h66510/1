import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
import matplotlib.pyplot as plt

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from torch.cuda.amp import autocast, GradScaler


# -----------------------------
# 1. 自定义 ResNet50 模型定义
# -----------------------------
class Bottleneck(nn.Module):
    # ResNet50 的 Bottleneck 结构 (expansion=4)
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64

        # 初始卷积层 (通常 ResNet 使用 7x7 卷积)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 四个残差层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 修改为适应你的分类任务 (例如疲劳检测通常是 2分类: 疲劳/非疲劳)
        # 注意：这里保留 num_classes 参数，训练时会被 get_model 函数覆盖
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class DrowsyDataset(torch.utils.data.Dataset):
    """疲劳驾驶数据集"""

    def __init__(self, data_dirs, class_name_file, transform=None):
        self.data_dirs = data_dirs if isinstance(data_dirs, list) else [data_dirs]
        self.transform = transform

        # 读取类别
        with open(class_name_file, 'r') as f:
            self.classes = [line.strip() for line in f if line.strip()]

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # 收集所有图片路径和标签
        self.samples = []
        for data_dir in self.data_dirs:
            for class_name in self.classes:
                class_dir = os.path.join(data_dir, class_name)
                if os.path.exists(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(class_dir, img_name)
                            self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_model(model_name, num_classes, pretrained=True):
    """获取模型 (集成自定义 ResNet50)"""
    if model_name == 'resnet50':
        # 使用我们自己搭建的 ResNet50
        # 注意：layers=[3, 4, 6, 3] 是 ResNet50 的标准配置
        model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 初始化 Scaler
    scaler = GradScaler()

    pbar = tqdm(dataloader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # 使用 autocast 上下文管理器
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # 使用 scaler 进行反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({'Loss': f'{running_loss / total:.4f}', 'Acc': f'{100. * correct / total:.2f}%'})
    return running_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validating')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({'Loss': f'{running_loss / total:.4f}', 'Acc': f'{100. * correct / total:.2f}%'})

    return running_loss / len(dataloader), 100. * correct / total


def get_transforms(input_size, mean, std, train=True):
    """获取数据转换"""
    if train:
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
def plot_curves(train_losses, train_accs, val_losses, val_accs, save_dir):
    """绘制并保存 Loss 和 Accuracy 曲线"""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Train Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'bD-', label='Train Accuracy')
    plt.plot(epochs, val_accs, 'rD-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # 保存图像
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=200)
    plt.close()  # 关闭图像释放内存
    logger.info(f"曲线图已保存至: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description='疲劳驾驶检测训练')
    parser.add_argument('-c', '--config', type=str, default='configs/config.yaml', help='配置文件路径')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r',encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # 创建输出目录
    os.makedirs(config['work_dir'], exist_ok=True)

    # 数据转换
    train_transform = get_transforms(
        config['input_size'],
        config['rgb_mean'],
        config['rgb_std'],
        train=True
    )
    test_transform = get_transforms(
        config['input_size'],
        config['rgb_mean'],
        config['rgb_std'],
        train=False
    )

    # 数据集
    train_dataset = DrowsyDataset(
        config['train_data'],
        config['class_name'],
        train_transform
    )
    test_dataset = DrowsyDataset(
        config['test_data'],
        config['class_name'],
        test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
    )

    logger.info(f'Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}')

    # 模型
    num_classes = len(train_dataset.classes)
    # 关键点：这里传入 'resnet50'，会调用上面我们定义的 ResNet 类
    model = get_model(config['net_type'], num_classes, config['pretrained'])
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config['lr'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config['milestones'],
        gamma=0.1
    )

    #用于存储指标的列表
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    # 训练循环
    best_acc = 0.0
    for epoch in range(config['num_epochs']):
        logger.info(f'Epoch {epoch + 1}/{config["num_epochs"]}')

        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        # 验证
        val_loss, val_acc = validate(model, test_loader, criterion, device)

        scheduler.step()

        # 记录指标
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            model_path = os.path.join(
                config['work_dir'],
                f'best_model_{epoch + 1:03d}_{val_acc:.4f}.pth'
            )
            torch.save(model.state_dict(), model_path)
            logger.info(f'Best model saved: {model_path}')

    logger.info('Training completed!')


if __name__ == '__main__':
    main()
