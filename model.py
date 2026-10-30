import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ResNet18Embedder(nn.Module):
    """ResNet-18 backbone projecting images to L2-normalized embeddings."""

    def __init__(self, embedding_dim=128, cifar_stem=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        backbone = resnet18(weights=None)

        if cifar_stem:
            backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            backbone.maxpool = nn.Identity()

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(512, embedding_dim)
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
        return F.normalize(x, p=2, dim=1, eps=1e-12)


class Optimizer:
    """Base optimizer class using raw tensor operations."""

    def __init__(self, params, lr=1e-3):
        self.params = [p for p in params if p.requires_grad]
        self.lr = float(lr)

    def zero_grad(self):
        with torch.no_grad():
            for p in self.params:
                if p.grad is not None:
                    p.grad.zero_()

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent with momentum implemented with tensor ops."""

    def __init__(self, params, lr=0.05, momentum=0.9, weight_decay=5e-4):
        super().__init__(params, lr=lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.velocity = [None for _ in self.params]

    def step(self):
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                g = p.grad.data
                if self.weight_decay != 0.0:
                    g = g + self.weight_decay * p.data
                if self.momentum != 0.0:
                    if self.velocity[i] is None:
                        self.velocity[i] = torch.zeros_like(p.data)
                    self.velocity[i].mul_(self.momentum).add_(g)
                    upd = self.velocity[i]
                else:
                    upd = g
                p.data.sub_(self.lr * upd)


class Adam(Optimizer):
    """Adam optimizer implemented from scratch with raw tensor operations."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4):
        super().__init__(params, lr=lr)
        self.beta1 = float(betas[0])
        self.beta2 = float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.t = 0
        self.m = [torch.zeros_like(p.data) for p in self.params]
        self.v = [torch.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        bc1 = 1.0 - (self.beta1 ** self.t)
        bc2 = 1.0 - (self.beta2 ** self.t)

        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                g = p.grad.data
                if self.weight_decay != 0.0:
                    g = g + self.weight_decay * p.data
                self.m[i].mul_(self.beta1).add_((1.0 - self.beta1) * g)
                self.v[i].mul_(self.beta2).add_((1.0 - self.beta2) * (g * g))
                m_hat = self.m[i] / bc1
                v_hat = self.v[i] / bc2
                p.data.sub_(self.lr * (m_hat / (torch.sqrt(v_hat) + self.eps)))


class NAdam(Optimizer):
    """NAdam optimizer implemented from scratch with raw tensor operations."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4):
        super().__init__(params, lr=lr)
        self.beta1 = float(betas[0])
        self.beta2 = float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.t = 0
        self.m = [torch.zeros_like(p.data) for p in self.params]
        self.v = [torch.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        bc1 = 1.0 - (self.beta1 ** self.t)
        bc2 = 1.0 - (self.beta2 ** self.t)

        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                g = p.grad.data
                if self.weight_decay != 0.0:
                    g = g + self.weight_decay * p.data
                self.m[i].mul_(self.beta1).add_((1.0 - self.beta1) * g)
                self.v[i].mul_(self.beta2).add_((1.0 - self.beta2) * (g * g))
                m_hat = self.m[i] / bc1
                v_hat = self.v[i] / bc2
                m_nest = self.beta1 * m_hat + ((1.0 - self.beta1) / bc1) * g
                p.data.sub_(self.lr * (m_nest / (torch.sqrt(v_hat) + self.eps)))


class CosineWarmupScheduler:
    """Cosine annealing learning rate schedule with linear warmup."""

    def __init__(self, optimizer, base_lr, total_epochs, warmup_epochs=3, min_lr=1e-5):
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.total_epochs = max(1, int(total_epochs))
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.min_lr = float(min_lr)

    def step(self, epoch):
        if epoch < self.warmup_epochs and self.warmup_epochs > 0:
            lr = self.base_lr * float(epoch + 1) / float(self.warmup_epochs)
        else:
            prog = float(epoch - self.warmup_epochs) / float(max(1, self.total_epochs - self.warmup_epochs))
            prog = min(max(prog, 0.0), 1.0)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * prog))
        self.optimizer.lr = lr
        return lr


ScratchSGD = SGD
ScratchAdam = Adam
ScratchNAdam = NAdam


def build_optimizer(model, optimizer_name="adam", lr=1e-3, weight_decay=1e-4, beta1=0.9, beta2=0.999, eps=1e-8, momentum=0.9):
    """Instantiate SGD, Adam, or NAdam optimizer."""
    name = optimizer_name.strip().lower()
    if name == "sgd":
        return SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return Adam(model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
    if name == "nadam":
        return NAdam(model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")
