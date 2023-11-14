import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import random
import scipy


class DecompTraceDataset(Dataset):
    """Documentation for DecompTraceDataset

    """

    def __init__(self,
                 root_dir: str = "",
                 target_length: int = 2048,
                 hflip: bool = False,
                 vflip: bool = False,
                 flip_rate: float = 0.1,
                 randshift: bool = False,
                 seed: int = 42,
                 sep_tok: bool = False,
                 normalize: bool = False,
                 num_classes: int = 1,
                 flat: bool = False):

        self.root_dir = root_dir
        self.target_length = target_length
        self.dir_list = os.listdir(self.root_dir)
        self.num_classes = num_classes

        # data augmentation flags
        self.hflip = hflip
        self.vflip = vflip
        self.randshift = randshift
        self.normal = normalize
        self.flip_rate = flip_rate
        self.flat = flat
        self.num_labels = len(self.dir_list)
        self.per_class_file_list = [None]*self.num_labels
        for i in range(len(self.per_class_file_list)):
            self.per_class_file_list[i] = []

        self.idx_array = []
        self.all_labels = []

        for i in range(0, self.num_labels):
            flist = os.listdir(os.path.join(self.root_dir, self.dir_list[i]))
            print("file list", len(flist))
            self.per_class_file_list[i] = flist

        self.num_individuals = [int]*self.num_labels
        for i in range(0, self.num_labels):
            self.num_individuals[i] = len(
                self.per_class_file_list[i])
            print(
                f"Number of total individual for class {self.dir_list[i]}: {self.num_individuals[i]}"
            )

        for i in range(0, self.num_labels):
            for j in range(0, self.num_individuals[i]):
                self.idx_array.append([i, j])
                self.all_labels.append(i)

        random.seed(seed)

    def __len__(self):
        total = 0
        for i in range(0, self.num_labels):
            total += self.num_individuals[i]
        return total

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fname = self.per_class_file_list[self.idx_array[idx]
                                         [0]][self.idx_array[idx][1]]
        path = os.path.join(
            self.root_dir, self.dir_list[self.idx_array[idx][0]]
        )

        fullpath = os.path.join(path, fname)

        data = np.loadtxt(
            fullpath,
            delimiter=",", dtype=float
        )

        label = self.idx_array[idx][0]

        """
        Sequence processing
        """
        seq = np.array(data)
        shape = seq.shape
        if self.normal:
            seq = seq.flatten()
            seq = scipy.stats.zscore(seq, axis=0)
            seq = seq.reshape(shape)

        if self.randshift:
            r = random.randint(0, 200)
        else:
            r = 100

        # print(seq[:, 500:2548].shape)seq[2:, 1000+r:4096+r]
        seq = seq[3:, 1500+r:3548+r]
        """seq = np.pad(seq[2:, 1000+r:4096+r], [(0, 0), (0, 1)],
                     mode='constant', constant_values=-1)  # .flatten()[:-1]"""

        # print(f"seq shape {seq.shape}")
        if self.flat:
            seq = seq.flatten()
            # print(f"seq shape {seq.shape}")
            mask = np.ones(len(seq))
            # print(f"mask {mask.shape}")
            # pad_size = self.target_length - seq.size
        else:
            mask = np.ones(seq.shape)

        if self.num_classes != 1:
            label = F.one_hot(
                torch.tensor(label, dtype=torch.int64), num_classes=self.num_labels)
            # print(f"labels shape {label.shape}")
            # label[(0, 1)[idx >= len(self.file_list_ano)]] = 1

        flip = np.random.random()
        invert = np.random.random()
        if self.hflip:
            if flip < self.flip_rate:
                seq = np.flip(seq)
        if self.vflip:
            if invert < self.flip_rate:
                seq = -seq

        return idx, seq[0:self.target_length].copy(), mask[0:self.target_length], label, len(data)
