#!/usr/bin/env python
# coding: utf-8




get_ipython().system('uv pip install medmnist')





from medmnist import INFO

# 关于OCTMNIST数据集的信息
info = INFO["octmnist"]

print("Dataset type: ", info["task"])
print("Dataset labels: ", info["label"])
print("Number of image channels: ", info["n_channels"])
print("Number of training samples: ", info["n_samples"]["train"])
print("Number of validation samples: ", info["n_samples"]["val"])
print("Number of test samples: ", info["n_samples"]["test"])





import torch

# 如果可用则使用GPU
# 注意：如果在Google Colab中运行联邦学习，请使用CPU

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")





from torchvision import transforms
from medmnist import OCTMNIST

# 定义数据转换
transform = transforms.ToTensor()

# Download datasets in size 64 x 64
train_dataset = OCTMNIST(split='train', transform=transform, download=True, size=64)
val_dataset = OCTMNIST(split='val', transform=transform, download=True, size=64)
test_dataset = OCTMNIST(split='test', transform=transform, download=True, size=64)





from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 定义标签映射
label_map = {
    0: 'choroidal neovascularization',
    1: 'diabetic macular edema',
    2: 'drusen',
    3: 'normal'
}

# Get a batch from the data loader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
images, labels = next(iter(train_loader))

# 绘图 in 3 x 3 grid
rows, cols = 3, 3
fig, axes = plt.subplots(rows, cols, figsize=(5, 5))

for i in range(rows * cols):
    ax = axes[i // cols, i % cols]
    ax.imshow(images[i][0], cmap='gray')
    ax.set_title(label_map[int(labels[i].item())], fontsize=6)
    ax.axis('off')

plt.tight_layout()
plt.show()





from torch.utils.data import Subset

# 创建子数据集
def create_sub_datasets(full_dataset):
    targets = torch.tensor([label.item() for _, label in full_dataset])
    mask_A = (targets == 0) | (targets == 2) | (targets == 3)
    mask_B = (targets == 0) | (targets == 1) | (targets == 3)
    mask_C = (targets == 1) | (targets == 2) | (targets == 3)

    indices_A = mask_A.nonzero(as_tuple=True)[0]
    indices_B = mask_B.nonzero(as_tuple=True)[0]
    indices_C = mask_C.nonzero(as_tuple=True)[0]

    dataset_A = Subset(train_dataset, indices_A)  # Contains: CNV, DRUSEN, NORMAL (excludes DME)
    dataset_B = Subset(train_dataset, indices_B)  # Contains: CNV, DME, NORMAL (excludes DRUSEN)
    dataset_C = Subset(train_dataset, indices_C)  # Contains: DME, DRUSEN, NORMAL (excludes CNV)

    return [dataset_A, dataset_B, dataset_C]

dataset_A, dataset_B, dataset_C = create_sub_datasets(train_dataset)





from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

# ResNet-18模型
def get_resnet_model(num_classes=4):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Modify first conv layer to accept 1 channel input
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Replace the final FC layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model.to(device)





import torch.optim as optim
from tqdm import tqdm

# 训练函数
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()

        running_correct, running_total = 0, 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)

        for images, labels in loop:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

            loop.set_postfix(loss=loss.item(), acc=running_correct / running_total)

        train_acc = running_correct / running_total

        val_acc = evaluate_model(model, val_loader)

        print(f"Epoch [{epoch+1}/{epochs}]  Train Acc: {train_acc:.4f}  Val Acc: {val_acc:.4f}")





# 评估函数
def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.squeeze().to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total





# 在数据子集上训练的函数
def train_on_subset(subset_dataset, val_loader, epochs=10):
    loader = DataLoader(subset_dataset, batch_size=64, shuffle=True)
    model = get_resnet_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, criterion, optimizer, loader, val_loader, epochs)

    return model





# 使用数据子集训练时的评估函数
def evaluate_on_test(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.squeeze().to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = sum([p == t for p, t in zip(all_preds, all_labels)]) / len(all_labels)

    return acc, all_preds, all_labels





# Train models on sub-datasets
val_loader = DataLoader(val_dataset, batch_size=64)

model_A = train_on_subset(dataset_A, val_loader)
model_B = train_on_subset(dataset_B, val_loader)
model_C = train_on_subset(dataset_C, val_loader)





# 在完整测试集上评估
test_loader = DataLoader(test_dataset, batch_size=64)

acc_A, preds_A, labels_A = evaluate_on_test(model_A, test_loader)
acc_B, preds_B, labels_B = evaluate_on_test(model_B, test_loader)
acc_C, preds_C, labels_C = evaluate_on_test(model_C, test_loader)

# 报告准确率
print(f"Test Accuracy | Model trained on dataset excluding DME: {acc_A:.4f}")
print(f"Test Accuracy | Model trained on dataset excluding DRUSEN: {acc_B:.4f}")
print(f"Test Accuracy | Model trained on dataset excluding CNV: {acc_C:.4f}")





import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["CNV", "DME", "DRUSEN", "NORMAL"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()





# 绘制混淆矩阵
plot_confusion_matrix(labels_A, preds_A, "Confusion Matrix - Model excluding DME")
plot_confusion_matrix(labels_B, preds_B, "Confusion Matrix - Model excluding DRUSEN")
plot_confusion_matrix(labels_C, preds_C, "Confusion Matrix - Model excluding CNV")


# ## 使用FlowerAI进行联邦学习




get_ipython().system('uv pip install "flwr[simulation]"')





# 获取训练后客户端模型的更新权重
def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]





from collections import OrderedDict

# 在训练前更新客户端模型的权重
def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)

    state_dict = OrderedDict({
        k: torch.tensor(v) for k, v in params_dict
    })

    net.load_state_dict(state_dict, strict=True)





from flwr.client import NumPyClient
from typing import Dict
from flwr.common import NDArrays, Scalar

# Flower客户端
class FlowerClient(NumPyClient):
    def __init__(self, net, trainset, valset, testset):
        self.net = net
        self.trainset = trainset
        self.valset = valset
        self.testset = testset

    # 本地训练
    def fit(self, parameters, config):
        set_weights(self.net, parameters)

        # Data loaders
        train_loader = DataLoader(self.trainset, batch_size=64, shuffle=True)
        val_loader = DataLoader(self.valset, batch_size=64)

        # Loss & Optimiser
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=0.001)

        train_model(
            self.net,
            criterion,
            optimizer,
            train_loader,
            val_loader,
            epochs= 1,
        )

        return get_weights(self.net), len(self.trainset), {}

    # 本地评估
    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, acc = evaluate_model(
            self.net, DataLoader(self.testset, batch_size=64)
        )

        return loss, len(self.testset), {"accuracy": acc}





from flwr.client import Client, ClientApp
from flwr.common import Context

train_sets = [dataset_A, dataset_B, dataset_C]

#  设置客户端的函数
def client_fn(context: Context) -> Client:
    cid = int(context.node_config["partition-id"])

    trainset = train_sets[cid]

    return FlowerClient(
        get_resnet_model(),
        trainset,
        val_dataset,
        test_dataset,
    ).to_client()

client = ClientApp(client_fn)





def filter_by_classes(dataset, class_list):
    indices = [i for i, (_, label) in enumerate(dataset) if label.item() in class_list]
    return Subset(dataset, indices)

# 包含：CNV、DRUSEN、NORMAL - 排除：DME
testset_no_dme = filter_by_classes(test_dataset, [0, 2, 3])

# 包含：CNV、DME、NORMAL - 排除：DRUSEN
testset_no_drusen = filter_by_classes(test_dataset, [0, 1, 3])

# 包含：DME、DRUSEN、NORMAL - 排除：CNV
testset_no_cnv = filter_by_classes(test_dataset, [1, 2, 3])





# 评估全局模型
def evaluate(server_round, parameters, config, num_rounds=5):
    net = get_resnet_model()
    set_weights(net, parameters)
    batch_size = 64

    acc_tot = evaluate_model(net, DataLoader(test_dataset, batch_size=batch_size))
    acc_A = evaluate_model(net, DataLoader(testset_no_dme, batch_size=batch_size))
    acc_B = evaluate_model(net, DataLoader(testset_no_drusen, batch_size=batch_size))
    acc_C = evaluate_model(net, DataLoader(testset_no_cnv, batch_size=batch_size))

    print(f"[Round {server_round}] Global  accuracy: {acc_tot:.4f}")
    print(f"[Round {server_round}] (CNV,DRUSEN,NORMAL) accuracy: {acc_A:.4f}")
    print(f"[Round {server_round}] (CNV,DME,NORMAL)    accuracy: {acc_B:.4f}")
    print(f"[Round {server_round}] (DME,DRUSEN,NORMAL) accuracy: {acc_C:.4f}")

    # 在最后一轮绘制混淆矩阵
    if server_round == num_rounds:
        acc_final, preds_final, labels_final = evaluate_on_test(net, DataLoader(test_dataset, batch_size=64))
        plot_confusion_matrix(labels_final, preds_final, "Final Global Confusion Matrix")





from flwr.common import ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

net = get_resnet_model()
params = ndarrays_to_parameters(get_weights(net))

# 设置全局服务器的函数
def server_fn(context: Context, num_rounds = 5):
    # 联邦平均策略
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        initial_parameters=params,
        evaluate_fn=evaluate,
    )

    config=ServerConfig(num_rounds)

    return ServerAppComponents(
        strategy=strategy,
        config=config,
    )

server = ServerApp(server_fn=server_fn)





from flwr.simulation import run_simulation
from logging import ERROR

# 保持日志输出简洁
backend_setup = {"init_args": {"logging_level": ERROR, "log_to_driver": False}}

# 运行训练模拟
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=3,
    backend_config=backend_setup,
)





# 检查类别不平衡

import matplotlib.pyplot as plt
from collections import Counter

# 统计类别出现次数
labels = [label.item() for _, label in train_dataset]
class_counts = Counter(labels)
print(class_counts)

# 准备绘图数据
class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
counts = [class_counts[i] for i in range(4)]

# 绘图
plt.figure(figsize=(8, 5))
plt.bar(class_names, counts)
plt.title("Class Distribution in OCTMNIST Training Set")
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()




