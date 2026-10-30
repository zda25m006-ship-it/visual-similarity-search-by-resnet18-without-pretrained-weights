import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision
from torchvision import transforms

MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)


def get_transforms(train=True, heavy=False):
    """Build image preprocessing and augmentation transforms for CIFAR-100."""
    if not train:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    ops = [
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
    if heavy:
        ops.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)))
    return transforms.Compose(ops)


def denormalize(tensor):
    """Convert a normalized CHW tensor back to an RGB uint8 numpy image."""
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.array(MEAN, dtype=np.float32)
    std = np.array(STD, dtype=np.float32)
    img = (img * std + mean) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


class CIFAR100Split(Dataset):
    """Dataset wrapper for a specific index split of CIFAR-100."""

    def __init__(self, images, labels, indices, class_names, transform=None):
        self.images = images[indices]
        self.labels = labels[indices].astype(np.int64)
        self.global_indices = indices
        self.class_names = class_names
        self.transform = transform
        self._to_pil = transforms.ToPILImage()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self._to_pil(self.images[idx])
        label = int(self.labels[idx])
        tensor = self.transform(img) if self.transform is not None else transforms.ToTensor()(img)
        return tensor, label, idx

    def get_raw_image(self, idx):
        return self.images[idx]


def split_indices(labels, train_ratio=0.80, val_ratio=0.10, test_ratio=0.10, seed=42):
    """Create deterministic per-class 80/10/10 splits with a fixed seed."""
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0)
    rng = np.random.RandomState(seed)
    num_classes = len(np.unique(labels))

    train_idx, val_idx, test_idx = [], [], []
    query_idx, gallery_idx = [], []

    for c in range(num_classes):
        c_idx = np.where(labels == c)[0]
        perm = rng.permutation(c_idx)
        n = len(perm)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))

        tr = perm[:n_train]
        va = perm[n_train:n_train + n_val]
        te = perm[n_train + n_val:]

        n_q = max(1, len(te) // 6)
        train_idx.append(tr)
        val_idx.append(va)
        test_idx.append(te)
        query_idx.append(te[:n_q])
        gallery_idx.append(te[n_q:])

    return {
        "train": np.concatenate(train_idx),
        "val": np.concatenate(val_idx),
        "test": np.concatenate(test_idx),
        "test_query": np.concatenate(query_idx),
        "test_gallery": np.concatenate(gallery_idx),
    }


class PKBatchSampler(Sampler):
    """Samples P classes and K instances per class for each mini-batch."""

    def __init__(self, labels, p_classes=32, k_samples=4, seed=42):
        self.labels = np.asarray(labels)
        self.p_classes = p_classes
        self.k_samples = k_samples
        self.batch_size = p_classes * k_samples
        self.seed = seed
        self.epoch = 0
        self.classes = np.unique(self.labels)
        self.class_map = {int(c): np.where(self.labels == c)[0] for c in self.classes}
        self.num_batches = len(self.labels) // self.batch_size

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        self.epoch += 1
        pools = {c: rng.permutation(idxs).tolist() for c, idxs in self.class_map.items()}

        for _ in range(self.num_batches):
            chosen = rng.choice(self.classes, size=min(self.p_classes, len(self.classes)), replace=False)
            batch = []
            for c in chosen:
                c_int = int(c)
                if len(pools[c_int]) < self.k_samples:
                    pools[c_int] = rng.permutation(self.class_map[c_int]).tolist()
                for _ in range(self.k_samples):
                    batch.append(pools[c_int].pop())
            yield batch

    def __len__(self):
        return self.num_batches


def build_cifar100_datasets(data_root="./data", seed=42, heavy_augment=False):
    """Load all 60,000 CIFAR-100 images and return train, val, and test splits."""
    tr_ds = torchvision.datasets.CIFAR100(root=data_root, train=True, download=True)
    te_ds = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True)

    images = np.concatenate([tr_ds.data, te_ds.data], axis=0)
    labels = np.concatenate([np.array(tr_ds.targets), np.array(te_ds.targets)], axis=0)
    class_names = list(tr_ds.classes)

    splits = split_indices(labels, 0.80, 0.10, 0.10, seed=seed)
    train_tf = get_transforms(train=True, heavy=heavy_augment)
    eval_tf = get_transforms(train=False)

    datasets = {
        "train": CIFAR100Split(images, labels, splits["train"], class_names, train_tf),
        "val": CIFAR100Split(images, labels, splits["val"], class_names, eval_tf),
        "test": CIFAR100Split(images, labels, splits["test"], class_names, eval_tf),
        "test_query": CIFAR100Split(images, labels, splits["test_query"], class_names, eval_tf),
        "test_gallery": CIFAR100Split(images, labels, splits["test_gallery"], class_names, eval_tf),
    }
    return datasets, class_names


def build_dataloaders(datasets, batch_size=128, k_samples=4, num_workers=2, seed=42, use_pk_sampler=True):
    """Create PyTorch DataLoaders for training, validation, and retrieval evaluation."""
    p_classes = max(2, batch_size // k_samples)

    if use_pk_sampler:
        tr_sampler = PKBatchSampler(datasets["train"].labels, p_classes=p_classes, k_samples=k_samples, seed=seed)
        train_loader = DataLoader(datasets["train"], batch_sampler=tr_sampler, num_workers=num_workers, pin_memory=True)
        va_sampler = PKBatchSampler(datasets["val"].labels, p_classes=p_classes, k_samples=k_samples, seed=seed + 100)
        val_triplet_loader = DataLoader(datasets["val"], batch_sampler=va_sampler, num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)
        val_triplet_loader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return {
        "train": train_loader,
        "val_triplet": val_triplet_loader,
        "val": DataLoader(datasets["val"], batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=True),
        "test": DataLoader(datasets["test"], batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=True),
        "test_query": DataLoader(datasets["test_query"], batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=True),
        "test_gallery": DataLoader(datasets["test_gallery"], batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=True),
    }
