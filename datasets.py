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
                (s, i) for i, (label, slices_i) in enumerate(zip(self.labels, slices))
                for s in slices_i if np.sum(label[s]) > 0
            ]
        else:
            self.patch_slices = [
                (s, i) for i, slices_i in enumerate(slices) for s in slices_i
            ]

    def __getitem__(self, index):
        # We select the case
        slice_i, case_idx = self.patch_slices[index]

        # We get the slice indexes
        none_slice = (slice(None, None),)

        inputs = self.data[case_idx][none_slice + slice_i].astype(np.float32)

        target = np.expand_dims(
            self.labels[case_idx][slice_i].astype(np.uint8), dim=0
        )

        # target_labs = bwlabeln(target.astype(np.bool))
        # tops = len(np.unique(target_labs[target.astype(np.bool)]))

        return inputs, target

    def __len__(self):
        return len(self.patch_slices)


class CroppingDown2DDataset(Cropping2DDataset):
    def __init__(
            self,
            data, labels, patch_size=32, overlap=16, filtered=False,
            ratio=10
    ):
        # Init
        downdata = [
            imresize(
                im, (im.shape[0],) + tuple(
                    [length // ratio for length in im.shape[1:]]
                ),
                order=2
            )
            for im in data
        ]
        downlabels = [
            torch.max_pool2d(
                torch.tensor(np.expand_dims(lab, 0)).type(torch.float32), ratio
            ).squeeze(dim=0).numpy().astype(np.bool)
            for lab in labels
        ]
        super().__init__(downdata, downlabels, patch_size, overlap, filtered)
