import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# 1. 设置Matplotlib使用支持中文的字体
plt.rcParams['font.family'] = 'SimHei'  # 使用SimHei或其他支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 用户设定的 n 值
# n_values = [0,-2,-4,-6,-8,-10]
n_values = [-6,-10]
# 数据预处理
data_transforms = transforms.Compose([
    transforms.ToTensor(),
])


# 函数：加载模型
def load_model(n, model_dir):
    model_path = os.path.join(model_dir, "resnet18_finetuned.pth")
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    # 假设有5个类别
    model.fc = torch.nn.Linear(num_ftrs, 5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


# 函数：加载数据
def load_data(val_dir, batch_size=32):
    dataset = datasets.ImageFolder(val_dir, data_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return dataloader, dataset.classes


# 函数：进行预测
def predict(model, dataloader):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds


# 函数：绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, classes, n, normalize=False):
    """
    绘制并保存混淆矩阵，使用中文类别名。

    参数：
    - y_true: 真实标签
    - y_pred: 预测标签
    - classes: 中文类别名称列表
    - n: 模型编号
    - normalize: 是否对混淆矩阵进行归一化（默认为 False）
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'  # 归一化时使用浮点数格式
    else:
        fmt = 'd'  # 非归一化时使用整数格式

    plt.figure(figsize=(20, 16))  # 增大图像尺寸以容纳更多类别
    sns.set(font_scale=1.2)  # 调整字体大小以增强可读性

    # 设置中文字体，确保可以显示中文字符
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 使用 seaborn 的 heatmap 绘制混淆矩阵
    heatmap = sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                          xticklabels=classes, yticklabels=classes,
                          annot_kws={"size": 10},  # 数字字体大小
                          linewidths=.5, linecolor='gray')

    # 设置标题和标签为中文
    plt.title(f'混淆矩阵 (jnr={n})', fontsize=20)
    plt.xlabel('预测类别', fontsize=16)  # "Predicted Class" in Chinese
    plt.ylabel('真实类别', fontsize=16)  # "True Class" in Chinese

    # 调整 x 轴标签的角度以避免重叠
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)

    # 定义 y 轴范围和刻度
    plt.ylim(len(classes), 0)  # 反转 y 轴以匹配矩阵
    plt.yticks(np.arange(len(classes)) + 0.5, classes, rotation=0, fontsize=12)
    plt.xticks(np.arange(len(classes)) + 0.5, classes, rotation=45, ha='right', fontsize=12)

    # 优化布局以防止标签被截断
    plt.tight_layout()

    # 保存混淆矩阵图像
    plt.savefig(f'confusion_matrix_n{n}.png', dpi=300)
    plt.close()

    print("混淆矩阵已成功保存为 PNG 文件。")


# 新函数：绘制聚合的 PRF 折线图
def plot_aggregated_prf(metrics_dict):
    """
    绘制多个模型的 Precision, Recall 和 F1-Score 的对比折线图。

    参数：
    - metrics_dict: 包含各模型 PRF 数据的字典，格式为 {n: {'Precision': p, 'Recall': r, 'F1-Score': f}, ...}
    """
    n_values_sorted = sorted(metrics_dict.keys())  # 确保n值按顺序排列
    precision = [metrics_dict[n]['Precision'] for n in n_values_sorted]
    recall = [metrics_dict[n]['Recall'] for n in n_values_sorted]
    f1 = [metrics_dict[n]['F1-Score'] for n in n_values_sorted]

    jnr_labels = [f'jnr={n}' for n in n_values_sorted]

    # 绘制 Precision 折线图
    plt.figure(figsize=(8, 6))
    plt.plot(jnr_labels, precision, marker='o', linestyle='-', color='b', label='Precision')
    plt.title('Precision Comparison Across Different jnr Values', fontsize=16)
    plt.xlabel('jnr Value', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.ylim(0, 1)
    for i, p in enumerate(precision):
        plt.text(i, p + 0.01, f"{p:.2f}", ha='center', va='bottom', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('precision_comparison.png', dpi=300)
    plt.close()

    # 绘制 Recall 折线图
    plt.figure(figsize=(8, 6))
    plt.plot(jnr_labels, recall, marker='o', linestyle='-', color='g', label='Recall')
    plt.title('Recall Comparison Across Different jnr Values', fontsize=16)
    plt.xlabel('jnr Value', fontsize=14)
    plt.ylabel('Recall', fontsize=14)
    plt.ylim(0, 1)
    for i, r in enumerate(recall):
        plt.text(i, r + 0.01, f"{r:.2f}", ha='center', va='bottom', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('recall_comparison.png', dpi=300)
    plt.close()

    # 绘制 F1-Score 折线图
    plt.figure(figsize=(8, 6))
    plt.plot(jnr_labels, f1, marker='o', linestyle='-', color='r', label='F1-Score')
    plt.title('F1-Score Comparison Across Different jnr Values', fontsize=16)
    plt.xlabel('jnr Value', fontsize=14)
    plt.ylabel('F1-Score', fontsize=14)
    plt.ylim(0, 1)
    for i, f in enumerate(f1):
        plt.text(i, f + 0.01, f"{f:.2f}", ha='center', va='bottom', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('f1_comparison.png', dpi=300)
    plt.close()


# 主流程
def main():
    # 字典用于存储各模型的PRF指标
    metrics_dict = {}

    for n in n_values:
        print(f"正在处理 n={n} ...")
        model_dir = f"jnr{n}"
        val_dir = os.path.join(model_dir, "val")

        # 检查模型文件和验证数据夹是否存在
        if not os.path.exists(model_dir):
            print(f"文件夹 {model_dir} 不存在，跳过。")
            continue
        model_path = os.path.join(model_dir, "resnet18_finetuned.pth")
        if not os.path.isfile(model_path):
            print(f"模型文件 {model_path} 不存在，跳过。")
            continue
        if not os.path.isdir(val_dir):
            print(f"验证数据夹 {val_dir} 不存在，跳过。")
            continue

        # 加载模型
        model = load_model(n, model_dir)

        # 加载数据
        dataloader, classes = load_data(val_dir)

        # 预测
        y_true, y_pred = predict(model, dataloader)

        # 打印分类报告
        print(f"分类报告 (n={n}):")
        print(classification_report(y_true, y_pred, target_names=classes))

        # 绘制混淆矩阵（如果需要）
        classes_chinese = ['多音干扰', '宽带干扰', '窄带干扰', '梳状谱干扰', '线性扫频干扰']
        plot_confusion_matrix(y_true, y_pred, classes_chinese, n)

        # 计算PRF指标并存储
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        metrics_dict[n] = {'Precision': precision, 'Recall': recall, 'F1-Score': f1}

        print(f"n={n} 处理完成。\n")

    # 绘制聚合的 PRF 折线图并保存CSV
    if metrics_dict:
        plot_aggregated_prf(metrics_dict)
        print("聚合的 PRF 折线图已保存为 'precision_comparison.png', 'recall_comparison.png', 'f1_comparison.png'.")

        # 将 metrics_dict 转换为 DataFrame 并保存为 CSV
        df = pd.DataFrame(metrics_dict).T  # 转置以便 jnr 为行
        df.to_csv('metrics.csv', index_label='jnr')
        print("PRF 数据已保存为 'metrics.csv'.")
    else:
        print("没有可绘制的指标。")



import matplotlib.pyplot as plt


def plot_metrics():
    """
    绘制 Precision、Recall 和 F1-Score 的折线图，并保存为 PNG 图片。

    该函数将:
    - 按照 jnr 值从小到大排序数据。
    - 设置纵轴范围为 0.9 到 1.05，每 0.1 显示一个刻度，且不显示 1.05。
    - 分别绘制 Precision、Recall 和 F1-Score 的折线图，并保存为 PNG 文件。
    """

    # 数据
    jnr = [0, -2, -4, -6, -8, -10]

    precision = [
        1.0,
        0.9970049251231281,
        0.9950345788347679,
        0.9811341473382201,
        0.9582761772640304,
        0.9049848159903042
    ]

    recall = [
        1.0,
        0.9970000000000001,
        0.9950000000000001,
        0.9809999999999999,
        0.958,
        0.9030000000000001
    ]

    f1_score = [
        1.0,
        0.9969999812498829,
        0.9949999125917998,
        0.9810113002615157,
        0.9578173976867683,
        0.9023545965901286
    ]

    # 将数据按照 jnr 从小到大排序
    sorted_data = sorted(zip(jnr, precision, recall, f1_score), key=lambda x: x[0])
    jnr_sorted, precision_sorted, recall_sorted, f1_score_sorted = zip(*sorted_data)

    # 转换为列表
    jnr_sorted = list(jnr_sorted)
    precision_sorted = list(precision_sorted)
    recall_sorted = list(recall_sorted)
    f1_score_sorted = list(f1_score_sorted)

    # 生成标签
    jnr_labels = [f'{n}' for n in jnr_sorted]

    # 设置图表的字体和风格（可选）
    plt.style.use('seaborn-darkgrid')  # 选择一个美观的样式

    # 定义 y 轴范围和刻度
    y_min = 0.9
    y_max = 1.005
    y_ticks = [x / 100 for x in range(90, 101)]

    # 定义一个内部函数来绘制单个图表
    def plot_single_metric(jnr_labels, metric_values, metric_name, color, marker, filename):
        plt.figure(figsize=(8, 6))
        plt.plot(jnr_labels, metric_values, marker=marker, linestyle='-', color=color, label=metric_name)
        # plt.title(f'{metric_name} Comparison Across Different jnr Values', fontsize=16)
        plt.xlabel('JNR', fontsize=14)
        plt.ylabel(metric_name, fontsize=14)
        plt.ylim(y_min, y_max)
        plt.yticks(y_ticks)
        for i, value in enumerate(metric_values):
            plt.text(i, value + 0.002, f"{value:.3f}", ha='center', va='bottom', fontsize=7)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    # 绘制 Precision 折线图
    plot_single_metric(
        jnr_labels=jnr_labels,
        metric_values=precision_sorted,
        metric_name='Precision',
        color='#6A5ACD',
        marker='o',
        filename='precision_comparison.png'
    )

    # 绘制 Recall 折线图
    plot_single_metric(
        jnr_labels=jnr_labels,
        metric_values=recall_sorted,
        metric_name='Recall',
        color='#FFA07A',
        marker='s',
        filename='recall_comparison.png'
    )

    # 绘制 F1-Score 折线图
    plot_single_metric(
        jnr_labels=jnr_labels,
        metric_values=f1_score_sorted,
        metric_name='F1-Score',
        color='#90EE90',
        marker='^',
        filename='f1_score_comparison.png'
    )

    print("所有图表已成功保存为 PNG 文件。")



if __name__ == "__main__":
    main()
    # plot_metrics()
