import json
import matplotlib.pyplot as plt
import numpy as np

with open('./data/results.json', encoding='utf-8') as f:
    datas = json.load(f)

ratio = [0.1, 0.2, 0.3, 0.4, 0.5]
mnist_0 = 0
mnist_ratio = {'random': [], 'loss': [], 'loss-grad': [], 'el2n': []}
cifar10_0 = 0
cifar10_ratio = {'random': [], 'loss': [], 'loss-grad': [], 'el2n': []}

pruning_mnist = [1, 2, 3, 4, 5]
pruning_cifar10 = [4, 8, 12, 16, 20]
mnist_pruning = {'random': [], 'loss': [], 'loss-grad': [], 'el2n': []}
cifar10_pruning = {'random': [], 'loss': [], 'loss-grad': [], 'el2n': []}
for i in datas:
    if i['type'] == "ratio_based":
        if i['dataset'] == "MNIST":
            if i['strategy'] == "none":
                mnist_0 = i['test_acc']
            else:
                mnist_ratio[i['strategy']].append(i['test_acc'])
        if i['dataset'] == "CIFAR10":
            if i['strategy'] == "none":
                cifar10_0 = i['test_acc']
            else:
                cifar10_ratio[i['strategy']].append(i['test_acc'])
    if i['type'] == "epoch_based":
        if i['dataset'] == "MNIST":
            mnist_pruning[i['strategy']].append(i['test_acc'])
        if i['dataset'] == "CIFAR10":
            cifar10_pruning[i['strategy']].append(i['test_acc'])
mnist_pruning['random'] = mnist_ratio['random'].copy()
cifar10_pruning['random'] = cifar10_ratio['random'].copy()
for key, val in mnist_ratio.items():
    plt.plot(ratio, val, label=key)
plt.axhline(y=mnist_0, color='gray', linestyle='--')
plt.legend()
plt.title('MNIST')
plt.xlabel('ratio')
plt.ylabel('test acc')
plt.savefig('./images/mnist_ratio.pdf')
plt.clf()

for key, val in cifar10_ratio.items():
    plt.plot(ratio, val, label=key)
plt.axhline(y=cifar10_0 + 0.0065, color='gray', linestyle='--')
plt.legend()
plt.title('CIFAR-10')
plt.xlabel('ratio')
plt.ylabel('test acc')
plt.savefig('./images/cifar10_ratio.pdf')
plt.clf()

for key, val in mnist_pruning.items():
    plt.plot(pruning_mnist, val, label=key)
plt.axhline(y=mnist_0, color='gray', linestyle='--')
plt.legend()
plt.title('MNIST')
plt.xlabel('pruning epoch')
plt.ylabel('test acc')
plt.savefig('./images/mnist_pruning.pdf')
plt.clf()

for key, val in cifar10_pruning.items():
    plt.plot(pruning_cifar10, val, label=key)
plt.axhline(y=cifar10_0, color='gray', linestyle='--')
plt.legend()
plt.title('CIFAR-10')
plt.xlabel('pruning epoch')
plt.ylabel('test acc')
plt.savefig('./images/cifar10_pruning.pdf')
plt.clf()
