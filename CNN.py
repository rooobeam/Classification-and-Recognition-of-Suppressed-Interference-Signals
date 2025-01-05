import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
# from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
from torchvision import datasets, transforms

def load_npy_as_tensor(path):
    """
    自定义加载函数，将 .npy 文件转换为 3 通道的张量。
    """
    # 加载 .npy 文件
    data = np.load(path).astype(np.float32)  # 假设形状为 (224, 224)

    # 转换为 torch 张量
    tensor = torch.from_numpy(data)  # 形状: (224, 224)

    # 添加通道维度并复制三通道
    tensor = tensor.unsqueeze(0).repeat(3, 1, 1)  # 形状: (3, 224, 224)

    return tensor


def main(k):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据预处理和增强
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            # 可选：添加归一化
        ]),
    }

    # 选择要训练的 jnr{k} 目录
    # k = '0'  # 替换为所需的k值，如 '0', '-2', '-4', '-6', '-8', '-10'
    data_dir = os.path.join('.', f'jnr{k}')  # 例如 './jnr0'

    # 检查数据目录是否存在
    if not os.path.isdir(data_dir):
        print(f"Error: {data_dir} 不存在。")
        return

    # 使用 ImageFolder 加载 .png 文件
    image_datasets = {
        x: datasets.ImageFolder(
            root=os.path.join(data_dir, x),
            transform=data_transforms[x]
        )
        for x in ['train', 'val']
    }

    # 数据加载器
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=32,
                      shuffle=True if x == 'train' else False, num_workers=4)
        for x in ['train', 'val']
    }

    # 获取数据集大小和类别名称
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print(f"类别数量: {num_classes}")
    print(f"类别名称: {class_names}")

    # 加载预训练的 ResNet-18 模型
    model = models.resnet18(pretrained=True)

    # 替换全连接层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # 将模型移动到设备
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    # 学习率调整策略（可选）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 训练和验证的函数
    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print("-" * 10)

            # 每个 epoch 包含训练和验证阶段
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # 设置为训练模式
                else:
                    model.eval()  # 设置为验证模式

                running_loss = 0.0
                running_corrects = 0

                # 迭代数据
                for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase}"):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # 前向传播
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # 反向传播和优化只在训练阶段
                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    # 统计
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            print()

        print("训练完成")
        return model

    # 开始训练
    trained_model = train_model(model, criterion, optimizer, scheduler, num_epochs=7)

    # 保存模型
    file_dir = os.path.join(data_dir, 'resnet18_finetuned.pth')  # 例如 './jnr0'
    torch.save(trained_model.state_dict(), file_dir)
    print("模型已保存为 resnet18_finetuned.pth")


if __name__ == '__main__':
    for k in ['0', '-2', '-4', '-6', '-8', '-10']:
        main(k)
