import torch.utils.data
import torchvision
import os
import numpy as np
import bz2
import scipy.io
from torchvision import transforms
from PIL import Image


class CombiningDataset(torch.utils.data.Dataset):

    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])

    def __getitem__(self, index):
        dataset_idx = 0
        cur = 0
        while cur + len(self.datasets[dataset_idx]) <= index:
            cur += len(self.datasets[dataset_idx])
            dataset_idx += 1
        item_idx = index - cur
        return self.datasets[dataset_idx].__getitem__(item_idx)


class _BaseDataset(torch.utils.data.Dataset):

    def __init__(self, root, is_train=True, transform=None):
        super().__init__()
        self.root = root
        self.is_train = is_train
        self.transform = transform
        self.images = []
        self.labels = []
        self.extract_images_labels()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        x = Image.fromarray(self.images[index])
        y = int(self.labels[index])
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def extract_images_labels(self):
        raise NotImplementedError


class MNIST(_BaseDataset):

    def __init__(self, root, is_train, transform):
        super().__init__(root, is_train, transform)

    def extract_images_labels(self):
        dataset = torchvision.datasets.MNIST(self.root, train=self.is_train, download=True)
        self.images, self.labels = [], []

        for i in range(10000 if self.is_train else len(dataset)):
            image, label = dataset[i]
            image = np.array(image)
            self.images.append(np.stack([image, image, image], axis=-1))
            self.labels.append(label)
        self.images, self.labels = np.stack(self.images), np.array(self.labels)


class SVHN(_BaseDataset):

    def __init__(self, root, is_train, transform):
        super().__init__(root, is_train, transform)

    def extract_images_labels(self):
        dataset = torchvision.datasets.SVHN(self.root, split='train' if self.is_train else 'test',
                                            download=True)
        self.images, self.labels = [], []

        for i in range(len(dataset)):
            image, label = dataset[i]
            self.images.append(np.array(image))
            self.labels.append(label)
        self.images, self.labels = np.stack(self.images), np.array(self.labels)


class USPS(_BaseDataset):

    def __init__(self, root, is_train, transform):
        super().__init__(root, is_train, transform)

    def extract_images_labels(self):
        if self.is_train:
            path = os.path.join(self.root, 'usps.bz2')
        else:
            path = os.path.join(self.root, 'usps.t.bz2')

        with bz2.BZ2File(path) as fp:
            raw_data = [line.decode().split() for line in fp.readlines()]
            images = [[x.split(':')[-1] for x in data[1:]] for data in raw_data]
            images = np.asarray(images, dtype=np.float32).reshape((-1, 16, 16))
            images = ((images + 1) / 2 * 255).astype(dtype=np.uint8)
            targets = [int(d[0]) - 1 for d in raw_data]

        self.images, self.labels = [], []
        for img, target in zip(images, targets):
            self.images.append(np.stack([img, img, img], axis=-1))
            self.labels.append(target)
        self.images, self.labels = np.stack(self.images), np.array(self.labels)


class SYN(_BaseDataset):

    def __init__(self, root, is_train, transform):
        super().__init__(root, is_train, transform)

    def extract_images_labels(self):
        if self.is_train:
            path = os.path.join(self.root, 'synth_train_32x32.mat')
        else:
            path = os.path.join(self.root, 'synth_test_32x32.mat')

        raw_data = scipy.io.loadmat(path)
        self.images = np.transpose(raw_data['X'], [3, 0, 1, 2])
        self.labels = raw_data['y'].reshape(-1)
        self.labels[np.where(self.labels == 10)] = 0
        self.labels = self.labels.astype(np.int64)


class MNIST_M(_BaseDataset):

    def __init__(self, root, is_train, transform):
        super().__init__(root, is_train, transform)

    def extract_images_labels(self):
        split_list = {
            'train': [
                "mnist_m_train",
                "mnist_m_train_labels.txt"
            ],
            'test': [
                "mnist_m_test",
                "mnist_m_test_labels.txt"
            ],
        }
        split = 'train' if self.is_train else 'test'
        data_dir, filename = split_list[split]
        full_path = os.path.join(self.root, filename)
        data_dir = os.path.join(self.root, data_dir)
        with open(full_path) as f:
            lines = f.readlines()

        lines = [line.split('\n')[0] for line in lines]
        files = [line.split(' ')[0] for line in lines]
        self.labels = np.array([int(line.split(' ')[1]) for line in lines]).reshape(-1)
        self.images = []
        for img in files:
            img = Image.open(os.path.join(data_dir, img)).convert('RGB')
            self.images.append(img)
        self.images = np.stack(self.images)


def get_datasets(task, root, domains, is_train):
    if task == 'PACS':
        if is_train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop((224, 224), (224, 224)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        datasets = [torchvision.datasets.ImageFolder(os.path.join(root, domain), transform=transform)
                    for domain in domains.split(',')]
    elif task == 'Digits':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor()
        ])
        datasets = [globals()[domain](os.path.join(root, domain), is_train, transform)
                    for domain in domains.split(',')]
    elif task == 'CIFAR10-C':
        preprocess = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5] * 3, [0.5] * 3)])
        if is_train:
            transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                 transforms.RandomCrop(32, padding=4),
                 preprocess])
            datasets = [torchvision.datasets.CIFAR10(root, train=is_train, transform=transform, download=True)]
        else:
            CORRUPTIONS = [
                ['gaussian_noise', 'shot_noise', 'impulse_noise'],
                ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'],
                ['snow', 'frost', 'fog', 'brightness'],
                ['contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
            ]
            transform = preprocess
            datasets = [torchvision.datasets.CIFAR10(root, train=is_train, transform=transform, download=True)
                        for _ in CORRUPTIONS * 5]
            all_data = []
            all_label = []
            label = np.load(os.path.join(root, 'CIFAR-10-C', 'labels.npy')).reshape((5, 10000))
            for corruptions in CORRUPTIONS:
                data = []
                for corruption in corruptions:
                    x = np.load(os.path.join(root, 'CIFAR-10-C', corruption + '.npy')).reshape((5, 10000, 32, 32, 3))
                    data.append(x)
                all_data.append(np.concatenate(data, axis=1))
                all_label.append(np.concatenate([label for _ in corruptions], axis=1))
            for idx, dataset in enumerate(datasets):
                level_idx = idx // 4
                domain_idx = idx % 4
                dataset.data = all_data[domain_idx][level_idx]
                dataset.targets = torch.LongTensor(all_label[domain_idx][level_idx])
    else:
        raise NotImplementedError
    return CombiningDataset(datasets) if is_train else datasets
