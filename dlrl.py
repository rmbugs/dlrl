#!/usr/bin/env python3
import argparse
import logging
import random
import numpy as np
from os import makedirs
from typing import Type
import time

import torch.backends.cudnn
import torchvision.transforms
from torch.func import functional_call, vmap, grad
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import transforms

from models import *

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置数据集保存路径
DATA_DIR = './datasets'
makedirs(DATA_DIR, exist_ok=True)

def indexed(cls: Type[torch.utils.data.Dataset]):
    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index
    return type(cls.__name__, (cls,), dict(__getitem__=__getitem__))

def get_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_train = indexed(MNIST)(root=DATA_DIR, train=True, download=True, transform=transform)
    mnist_test = indexed(MNIST)(root=DATA_DIR, train=False, download=True, transform=transform)
    return mnist_train, mnist_test

def get_cifar10():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    cifar_train = indexed(CIFAR10)(root=DATA_DIR, train=True, download=True, transform=transform_train)
    cifar_test = indexed(CIFAR10)(root=DATA_DIR, train=False, download=True, transform=transform_test)
    return cifar_train, cifar_test

class Config:
    def __init__(self,
                 dataset: str,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 batch_size: int = 64,
                 num_workers: int = 2):
        self.train_set, self.test_set = self.get_dataset(dataset)
        self.device = torch.device(device)
        self.model = self.get_model(dataset).to(self.device)
        if self.device.type == 'cuda':
            self.model = nn.DataParallel(self.model)
        self.batch_size = batch_size
        self.num_workers = num_workers

    @staticmethod
    def get_dataset(dataset: str):
        if dataset == 'MNIST':
            return get_mnist()
        elif dataset == 'CIFAR10':
            return get_cifar10()
        else:
            raise ValueError(f"Invalid dataset: {dataset}")

    @staticmethod
    def get_model(dataset: str):
        if dataset == 'MNIST':
            return MNISTNet()
        elif dataset == 'CIFAR10':
            return ResNet18()
        else:
            raise ValueError(f"Invalid dataset: {dataset}")

def parse_args():
    parser = argparse.ArgumentParser(
        description='Deep Learning Dataset Pruning Experiment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # use different default parameters for different datasets
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10'], help='dataset to use')

    # basic
    dataset = parser.parse_known_args()[0].dataset
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='optimizer type')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       choices=['cuda', 'cpu'], help='device to use for training')
    parser.add_argument('--batch-size', type=int, default=128, help='number of samples per batch')
    parser.add_argument('--num-workers', type=int, default=4, help='number of worker processes for data loading')

    # training
    default_params = {
        'MNIST': {
            'lr': 0.1,
            'lr_milestones': [10, 20],
            'lr_gamma': 0.2,
            'weight_decay': 5e-4,
            'epochs': 30,
            'pruning_epoch': 3
        },
        'CIFAR10': {
            'lr': 0.1,
            'lr_milestones': [60, 120, 160],
            'lr_gamma': 0.2,
            'weight_decay': 5e-4,
            'epochs': 200,
            'pruning_epoch': 20
        }
    }
    params = default_params[dataset]
    parser.add_argument('--epochs', type=int, default=params['epochs'],
                       help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=params['lr'], help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=params['weight_decay'], help='weight decay for optimizer')

    # pruning
    parser.add_argument('--pruning-strategy', type=str, default='random', choices=['random', 'loss', 'loss-grad', 'el2n'])
    parser.add_argument('--pruning-ratio', type=float, default=0.5, help='ratio of data to prune, in range [0, 1]')
    parser.add_argument('--pruning-epoch', type=int, default=params['pruning_epoch'], help='epoch at which to perform pruning')

    return parser.parse_args()

class Trainer:
    def __init__(self, optimizer_class, model: nn.Module, device: torch.device, lr: float = 0.1, weight_decay: float = 5e-4):
        self.model = model
        if optimizer_class == SGD:
            self.optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 20], gamma=0.2)
        elif optimizer_class == Adam:
            self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        else:
            raise ValueError(f"Invalid optimizer: {optimizer_class}")
        self.device = device


    def train(self, dataloader):
        self.model.train()
        running_loss = 0

        for data, target, _ in dataloader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * data.size(0)

        return running_loss / len(dataloader.dataset)

    def test_acc(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            for data, target, _ in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
        return correct / len(dataloader.dataset)

class PrunerContext:
    def __init__(self, config: Config, dataloader):
        self.model = config.model.module if isinstance(config.model, nn.DataParallel) else config.model
        self.params = {k: v.detach() for k, v in self.model.named_parameters()}
        self.buffers = {k: v.detach() for k, v in self.model.named_buffers()}
        self.device = config.device
        self.num_samples = len(dataloader.dataset)
        self.dataloader = dataloader

    def _loss_func(self, params, buffers, data, target):
        """Returns a function call to compute loss for a single sample."""
        batch = data.unsqueeze(0)
        targets = target.unsqueeze(0)

        predictions = functional_call(self.model, (params, buffers), (batch,))
        loss = F.cross_entropy(predictions, targets)
        return loss

    def compute_loss(self):
        """Returns a vector of loss for each sample."""
        self.model.eval()
        losses = torch.zeros(self.num_samples, device='cpu')
        ft_compute_sample_losses = vmap(self._loss_func, in_dims=(None, None, 0, 0))

        for data, target, indices in self.dataloader:
            data, target = data.to(self.device), target.to(self.device)
            losses[indices] = ft_compute_sample_losses(self.params, self.buffers, data, target).cpu()

        self.model.train()
        return losses

    def compute_loss_grad(self):
        """Returns a vector of loss gradient for each sample."""
        self.model.eval()
        loss_grads = torch.zeros(self.num_samples, device='cpu')
        ft_compute_sample_grads = vmap(grad(self._loss_func), in_dims=(None, None, 0, 0))

        for data, target, indices in self.dataloader:
            data, target = data.to(self.device), target.to(self.device)
            loss_grad = ft_compute_sample_grads(self.params, self.buffers, data, target).values() # iterator of gradients with shape (batch_size, ...)
            squared_norm = torch.sum(torch.stack([torch.norm(g.view(g.size(0), -1), p=2, dim=1) ** 2 for g in loss_grad]), dim=0)
            loss_grads[indices] = torch.sqrt(squared_norm).cpu()

        self.model.train()
        return loss_grads

    def compute_el2n(self):
        """Returns a vector of el2n for each sample."""
        self.model.eval()
        el2n = torch.zeros(self.num_samples, device='cpu')

        with torch.no_grad():
            for data, target, indices in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                probs = F.softmax(outputs, dim=1)
                probs[torch.arange(probs.size(0), device=self.device), target] -= 1 # calculate error vector in-place
                el2n[indices] = torch.norm(probs, dim=1, p=2).cpu()
                print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB\n")

        self.model.train()
        return el2n

class Pruner:
    def __init__(self, strategy: str):
        self.strategy = strategy

    def get_sample_score(self, ctx: PrunerContext):
        if self.strategy == 'random':
            return torch.rand(ctx.num_samples)
        elif self.strategy == 'loss':
            return ctx.compute_loss()
        elif self.strategy == 'loss-grad':
            return ctx.compute_loss_grad()
        elif self.strategy == 'el2n':
            return ctx.compute_el2n()
        assert 0

    def select_samples(self, ctx: PrunerContext, pruning_ratio):
        """Return indices of samples to keep."""
        num_samples = ctx.num_samples
        num_keep = int(num_samples * (1 - pruning_ratio))
        scores = self.get_sample_score(ctx)
        return torch.argsort(scores, descending=True)[:num_keep]

def main():
    args = parse_args()
    config = Config(
        dataset=args.dataset,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    if args.optimizer.lower() == 'sgd':
        optimizer_class = SGD
    elif args.optimizer.lower() == 'adam':
        optimizer_class = Adam
    else:
        raise ValueError(f"Invalid optimizer: {args.optimizer}")

    if args.pruning_ratio > 0.0:
        logger.info("Computing scores for pruning...")
        pruner = Pruner(args.pruning_strategy)
        trainer = Trainer(
            optimizer_class=optimizer_class,
            model=config.model,
            device=config.device,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        train_loader = DataLoader(
            config.train_set,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=True
        )
        for epoch in range(args.pruning_epoch):
            train_loss = trainer.train(train_loader)
            logger.info("Epoch %d, Train Loss: %.6f", epoch, train_loss)
            trainer.scheduler.step()
        train_loader = DataLoader(
            config.train_set,
            batch_size=32,
            num_workers=config.num_workers,
            shuffle=True
        )
        ctx = PrunerContext(config, train_loader)
        indices = pruner.select_samples(ctx, args.pruning_ratio)
        config.train_set = torch.utils.data.Subset(config.train_set, indices)
        logger.info("Remaining samples: %d", len(config.train_set))

    # Create CUDA events for timing
    if config.device.type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
    else:
        start_time = time.time()

    logger.info("Training...")
    trainer = Trainer(
        optimizer_class=optimizer_class,
        model=config.model,
        device=config.device,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    train_loader = DataLoader(
        config.train_set,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True
    )
    test_loader = DataLoader(
        config.test_set,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    for epoch in range(args.epochs):
        train_loss = trainer.train(train_loader)
        logger.info("Epoch %d, Train Loss: %.6f", epoch, train_loss)
        trainer.scheduler.step()

    test_acc = trainer.test_acc(test_loader)
    logger.info("Final Test Acc: %.6f", test_acc)

    if config.device.type == 'cuda':
        end_event.record()
        torch.cuda.synchronize()  # Wait for all GPU operations to finish
        total_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
    else:
        total_time = time.time() - start_time

    logger.info("Total training time: %.2f seconds", total_time)

if __name__ == "__main__":
    main()
