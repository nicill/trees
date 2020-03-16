from skimage.transform import resize as imresize
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from data_manipulation.datasets import get_slices_bb


class Cropping2DDataset(Dataset):
    def __init__(
            self,
            data, labels, patch_size=32, overlap=16, filtered=False
    ):
        # Init
        self.data = data
        self.labels = labels
        data_shape = self.data[0].shape

        if type(patch_size) is not tuple:
            patch_size = (patch_size,) * len(data_shape)
        self.patch_size = patch_size
        self.overlap = overlap

        slices = get_slices_bb(
            self.labels, self.patch_size, self.overlap
        )
        if filtered:
            self.patch_slices = [
                [s for s in slices_i if np.sum(label[s]) > 0]
                for label, slices_i in zip(self.labels, slices)
            ]
        else:
            self.patch_slices = slices

        self.max_slice = np.cumsum(list(map(len, self.patch_slices)))

    def __getitem__(self, index):
        # We select the case
        case_idx = np.min(np.where(self.max_slice > index))

        slices = [0] + self.max_slice.tolist()
        patch_idx = index - slices[case_idx]
        case_slices = self.patch_slices[case_idx]

        # We get the slice indexes
        none_slice = (slice(None, None),)
        slice_i = case_slices[patch_idx]

        inputs = self.data[case_idx][none_slice + slice_i].astype(np.float32)

        labels = self.labels[case_idx][slice_i].astype(np.uint8)
        target = np.expand_dims(labels, 0)

        return inputs, target

    def __len__(self):
        return self.max_slice[-1]


class CroppingDown2DDataset(Dataset):
    def __init__(
            self,
            data, labels, patch_size=32, overlap=16, filtered=False,
            ratio=10
    ):
        # Init
        data_shape = data[0].shape
        dim = len(data_shape)
        self.data = [
            imresize(
                im, (im.shape[0],) + tuple(
                    [length // ratio for length in im.shape[1:]]
                ),
                order=2
            )
            for im in data
        ]
        self.labels = [
            torch.max_pool2d(
                torch.tensor(np.expand_dims(lab, 0)).type(torch.float32), ratio
            ).numpy().astype(np.bool)
            for lab in labels
        ]

        if type(patch_size) is not tuple:
            patch_size = (patch_size,) * dim
        self.patch_size = patch_size
        self.overlap = overlap

        slices = get_slices_bb(
            [np.squeeze(lab) for lab in self.labels],
            self.patch_size, self.overlap
        )
        if filtered:
            self.patch_slices = [
                [s for s in slices_i if np.sum(
                    label[(slice(None, None),) + s]
                ) > 0]
                for label, slices_i in zip(self.labels, slices)
            ]
        else:
            self.patch_slices = slices

        self.max_slice = np.cumsum(list(map(len, self.patch_slices)))
        print(
            np.cumsum(list(map(len, slices))),
            self.max_slice
        )

    def __getitem__(self, index):
        # We select the case
        case_idx = np.min(np.where(self.max_slice > index))

        slices = [0] + self.max_slice.tolist()
        patch_idx = index - slices[case_idx]
        case_slices = self.patch_slices[case_idx]

        # We get the slice indexes
        none_slice = (slice(None, None),)
        slice_i = case_slices[patch_idx]

        inputs = self.data[case_idx][none_slice + slice_i].astype(np.float32)

        target = self.labels[case_idx][none_slice + slice_i].astype(np.uint8)

        return inputs, target

    def __len__(self):
        return self.max_slice[-1]
