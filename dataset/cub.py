import os
import json
import pickle
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CUB(Dataset):
    def __init__(self,
                 args,
                 partition='train',
                 pretrain=True,
                 is_sample=False,
                 k=4096,
                 transform=None):
        super(Dataset, self).__init__()
        self.data_root = args.data_root
        self.partition = partition
        self.data_aug = args.data_aug
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.pretrain = pretrain

        if transform is None:
            if self.partition == 'train' and self.data_aug:
                self.transform = transforms.Compose([
                    # lambda x: Image.fromarray(x),
                    transforms.RandomCrop(224, padding=16),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    self.normalize
                ])
            else:
                self.transform = transforms.Compose([
                    # lambda x: Image.fromarray(x),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    self.normalize
                ])
        else:
            self.transform = transform

        if self.pretrain:
            self.file_pattern = 'cub_category_split_train_phase_%s.pickle'
        else:
            self.file_pattern = 'cub_category_split_%s.pickle'
        self.data = {}
        with open(os.path.join(self.data_root, self.file_pattern % partition), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            self.imgs = data['data']
            self.labels = data['labels'].astype(np.int64)

        # pre-process for contrastive sampling
        self.k = k
        self.is_sample = is_sample
        if self.is_sample:
            self.labels = np.asarray(self.labels)
            self.labels = self.labels - np.min(self.labels)
            num_classes = np.max(self.labels) + 1

            self.cls_positive = [[] for _ in range(num_classes)]
            for i in range(len(self.imgs)):
                self.cls_positive[self.labels[i]].append(i)

            self.cls_negative = [[] for _ in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]
            self.cls_positive = np.asarray(self.cls_positive)
            self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, item):
        img = np.asarray(self.imgs[item]).astype('uint8')
        img = Image.fromarray(img)
        img = self.transform(img)
        target = self.labels[item] - min(self.labels)

        if not self.is_sample:
            return img, target, item
        else:
            pos_idx = item
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, item, sample_idx

    def __len__(self):
        return len(self.labels)


class MetaCUB(CUB):

    def __init__(self,
                 args,
                 partition='train',
                 train_transform=None,
                 test_transform=None,
                 fix_seed=True):
        super(MetaCUB, self).__init__(args, partition=partition, pretrain=False)
        self.fix_seed = fix_seed
        self.n_ways = args.n_ways
        self.n_shots = args.n_shots
        self.n_queries = args.n_queries
        self.classes = list(self.data.keys())
        self.n_test_runs = args.n_test_runs
        self.n_aug_support_samples = args.n_aug_support_samples
        if train_transform is None:
            self.train_transform = transforms.Compose([
                # lambda x: Image.fromarray(x),
                transforms.RandomCrop(224, padding=16),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.train_transform = train_transform

        if test_transform is None:
            self.test_transform = transforms.Compose([
                # lambda x: Image.fromarray(x),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.test_transform = test_transform

        self.data = {}
        for idx in range(self.imgs.shape[0]):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.imgs[idx])
        self.classes = list(self.data.keys())

    def __getitem__(self, item):
        if self.fix_seed:
            np.random.seed(item)
        cls_sampled = np.random.choice(self.classes, self.n_ways, False)
        support_xs = []
        support_ys = []
        query_xs = []
        query_ys = []
        for idx, cls in enumerate(cls_sampled):
            imgs = np.asarray(self.data[cls]).astype('uint8')
            support_xs_ids_sampled = np.random.choice(range(imgs.shape[0]), self.n_shots, False)
            support_xs.append(imgs[support_xs_ids_sampled])
            support_ys.append([idx] * self.n_shots)
            query_xs_ids = np.setxor1d(np.arange(imgs.shape[0]), support_xs_ids_sampled)
            query_xs_ids = np.random.choice(query_xs_ids, self.n_queries, False)
            query_xs.append(imgs[query_xs_ids])
            query_ys.append([idx] * query_xs_ids.shape[0])
        support_xs, support_ys, query_xs, query_ys = np.array(support_xs), np.array(support_ys), np.array(
            query_xs), np.array(query_ys)
        num_ways, n_queries_per_way, height, width, channel = query_xs.shape
        query_xs = query_xs.reshape((num_ways * n_queries_per_way, height, width, channel))
        query_ys = query_ys.reshape((num_ways * n_queries_per_way,))

        support_xs = support_xs.reshape((-1, height, width, channel))
        if self.n_aug_support_samples > 1:
            support_xs = np.tile(support_xs, (self.n_aug_support_samples, 1, 1, 1))
            support_ys = np.tile(support_ys.reshape((-1,)), (self.n_aug_support_samples))
        support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
        query_xs = query_xs.reshape((-1, height, width, channel))
        query_xs = np.split(query_xs, query_xs.shape[0], axis=0)

        support_xs = torch.stack(list(map(lambda x: self.train_transform(Image.fromarray(x.squeeze())), support_xs)))
        query_xs = torch.stack(list(map(lambda x: self.test_transform(Image.fromarray(x.squeeze())), query_xs)))

        return support_xs, support_ys, query_xs, query_ys

    def __len__(self):
        return self.n_test_runs


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class CUBImageFolderDataset(Dataset):
    def __init__(self,
                 args,
                 transform=None,
                 target_transform=None,
                 train_val_test=0,
                 class_start_idx=0,
                 class_end_idx=199,
                 image_decoder=pil_loader):
        img_folder = os.path.join(args.data_root, "images")
        img_paths = pd.read_csv(os.path.join(args.data_root, "images.txt"), sep=" ", header=None, names=['idx', 'path'])
        img_labels = pd.read_csv(os.path.join(args.data_root, "image_class_labels.txt"), sep=" ", header=None,
                                 names=['idx', 'label'])
        data = pd.concat([img_paths, img_labels], axis=1)

        # split dataset
        data = data[data['label'] <= class_end_idx]
        data = data[data['label'] >= class_start_idx]
        data['label'] = data['label'] - class_start_idx

        imgs = data.reset_index(drop=True)

        if len(imgs) == 0:
            raise (RuntimeError("no csv file"))
        self.root = img_folder
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.image_decoder = image_decoder
        if train_val_test == 0:
            self.train = True
        else:
            self.train = False

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        item = self.imgs.iloc[int(index)]
        file_path = item['path']
        target = item['label']

        img = self.image_decoder(os.path.join(self.root, file_path))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class MetaCUBImageFolder(CUBImageFolderDataset):
    def __init__(self,
                 args,
                 train_transform=None,
                 test_transform=None,
                 class_start_idx=0,
                 class_end_idx=199,
                 fix_seed=True):
        super(MetaCUBImageFolder, self).__init__(args=args,
                                                 class_start_idx=class_start_idx,
                                                 class_end_idx=class_end_idx)
        self.fix_seed = fix_seed
        self.n_ways = args.n_ways
        self.n_shots = args.n_shots
        self.n_queries = args.n_queries
        self.n_test_runs = args.n_test_runs
        self.n_aug_support_samples = args.n_aug_support_samples
        if train_transform is None:
            self.train_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
        else:
            self.train_transform = train_transform

        if test_transform is None:
            self.test_transform = transforms.Compose(
                [
                    transforms.Resize((int(246), int(256))),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
        else:
            self.test_transform = test_transform

        self.classes = class_end_idx - class_start_idx + 1

    def __getitem__(self, item):
        if self.fix_seed:
            np.random.seed(item)
        cls_sampled = np.random.choice(self.classes, self.n_ways, False)
        support_xs = []
        support_ys = []
        query_xs = []
        query_ys = []
        for idx, cls in enumerate(cls_sampled):
            one_class_imgs = self.imgs[self.imgs['label'] == cls]
            one_class_imgs = one_class_imgs.reset_index(drop=True)
            support_xs_ids_sampled = np.random.choice(range(one_class_imgs.shape[0]), self.n_shots, False)
            query_xs_ids = np.setxor1d(np.arange(one_class_imgs.shape[0]), support_xs_ids_sampled)
            query_xs_ids = np.random.choice(query_xs_ids, self.n_queries, False)

            for support_id in support_xs_ids_sampled:
                item = one_class_imgs.iloc[int(support_id)]
                file_path = item['path']
                img = self.image_decoder(os.path.join(self.root, file_path))
                for _ in range(self.n_aug_support_samples):
                    temp_img = self.train_transform(img)
                    support_xs.append(temp_img)
            support_ys.append([idx] * self.n_shots * self.n_aug_support_samples)

            for query_id in query_xs_ids:
                item = one_class_imgs.iloc[int(query_id)]
                file_path = item['path']
                img = self.image_decoder(os.path.join(self.root, file_path))
                temp_img = self.test_transform(img)
                query_xs.append(temp_img)
            query_ys.append([idx] * self.n_queries)

        support_xs = torch.stack(support_xs)
        support_ys = torch.tensor(np.array(support_ys), dtype=torch.long)
        support_ys = support_ys.reshape(-1)
        query_xs = torch.stack(query_xs)
        query_ys = torch.tensor(np.array(query_ys), dtype=torch.long)
        query_ys = query_ys.reshape(-1)

        return support_xs, support_ys, query_xs, query_ys

    def __len__(self):
        return self.n_test_runs


if __name__ == '__main__':
    args = lambda x: None
    args.n_ways = 5
    args.n_shots = 1
    args.n_queries = 12
    args.data_root = json.load(open('config.json'))['CUB']
    args.data_aug = True
    args.n_test_runs = 5
    args.n_aug_support_samples = 1
    cub = CUB(args, partition='train')
    print(len(cub))
    print(cub.__getitem__(500)[0].shape)

    meta_cub = MetaCUB(args, partition='val')
    print(len(meta_cub))
    print(meta_cub.__getitem__(500)[0].size())
    print(meta_cub.__getitem__(500)[1].shape)
    print(meta_cub.__getitem__(500)[2].size())
    print(meta_cub.__getitem__(500)[3].shape)