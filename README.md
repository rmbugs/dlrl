# Deep Learning Dataset Pruning Experiments

这个项目实现了几种数据集剪枝(Dataset Pruning)策略，并在MNIST和CIFAR10数据集上进行了实验对比。

## 支持的剪枝策略

- Random: 随机选择样本
- Loss: 选择代理模型上损失值最大的样本
- Loss-Grad: 选择代理模型上损失梯度范数最大的样本
- EL2N: 选择代理模型上误差向量范数最大的样本

## 安装

1. 克隆仓库
2. 安装依赖

```bash
pip install -r requirements.txt
```

## 运行

运行单次实验示例:

```bash
python dlrl.py --dataset MNIST --pruning-strategy el2n --pruning-ratio 0.5
```

运行完整实验组合:
```bash
pytest test.py
```

查看所有实验，但不运行：
```bash
pytest test.py --dry-run
```

## 查看结果

所有结果会在测试过程中实时更新。
- `pytest.log` 训练脚本的输出
- `results.json` 所有实验结果

## 默认配置

### MNIST
- 模型: 简单CNN (2层卷积 + 2层全连接)
- 优化器: SGD
  - 学习率: 0.1
  - 权重衰减: 5e-4
  - 动量: 0.9
- 训练配置:
  - 训练轮次: 30
  - 代理模型训练轮次: 3
  - 学习率调度: 第 10, 20 轮衰减为 0.2 倍
  - 批次大小: 128

### CIFAR10
- 模型: ResNet18
- 优化器: SGD
  - 学习率: 0.1
  - 权重衰减: 5e-4
  - 动量: 0.9
- 训练配置:
  - 训练轮次: 200
  - 代理模型训练轮次: 20
  - 学习率调度: Cosine Annealing
  - 批次大小: 128
