import itertools
from skimage.transform import resize as imresize
import numpy as np
import torch
from torch.utils.data.dataset import Dataset


def centers_to_slice(voxels, patch_half):
    slices = [
        tuple(
            [
                slice(idx - p_len, idx + p_len) for idx, p_len in zip(
                    voxel, patch_half
                )
            ]
        ) for voxel in voxels
    ]
    return slices


def get_slices(masks, patch_size, overlap):
    """
    Function to get all the patches with a given patch size and overlap between
    consecutive patches from a given list of masks. We will only take patches
    inside the bounding box of the mask. We could probably just pass the shape
    because the masks should already be the bounding box.
    :param masks: List of masks.
    :param patch_size: Size of the patches.
    :param overlap: Overlap on each dimension between consecutive patches.

    """
    # Init
    # We will compute some intermediate stuff for later.
    patch_half = [p_length // 2 for p_length in patch_size]
    steps = [max(p_length - o, 1) for p_length, o in zip(patch_size, overlap)]

    # We will need to define the min and max pixel indices. We define the
    # centers for each patch, so the min and max should be defined by the
    # patch halves.
    min_bb = [patch_half] * len(masks)
    max_bb = [
        [
            max_i - p_len for max_i, p_len in zip(mask.shape, patch_half)
        ] for mask in masks
    ]

    # This is just a "pythonic" but complex way of defining all possible
    # indices given a min, max and step values for each dimension.
    dim_ranges = [
        [
            np.concatenate([np.arange(*t), [t[1]]])
            for t in zip(min_bb_i, max_bb_i, steps)
        ] for min_bb_i, max_bb_i in zip(min_bb, max_bb)
    ]

    # And this is another "pythonic" but not so intuitive way of computing
    # all possible triplets of center voxel indices given the previous
    # indices. I also added the slice computation (which makes the last step
    # of defining the patches).
    patch_slices = [
        centers_to_slice(
            itertools.product(*dim_range), patch_half
        ) for dim_range in dim_ranges
    ]

    return patch_slices


class Cropping2DDataset(Dataset):
    def __init__(
            self,
            data, labels, rois, patch_size=32, overlap=16, filtered=False
    ):
        # Init
        self.data = data
        self.labels = labels
        self.rois = rois
        data_shape = self.data[0].shape

        if type(patch_size) is not tuple:
            patch_size = (patch_size,) * len(data_shape)
        self.patch_size = patch_size
        self.overlap = overlap

        slices = get_slices(
            self.rois, self.patch_size, self.overlap
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

        inputs = (
            self.data[case_idx][none_slice + slice_i].astype(np.float32),
            np.expand_dims(
                self.rois[case_idx][slice_i].astype(np.uint8), axis=0
            )
        )

        target = np.expand_dims(
            self.labels[case_idx][slice_i].astype(np.uint8), axis=0
        )

        # target_labs = bwlabeln(target.astype(np.bool))
        # tops = len(np.unique(target_labs[target.astype(np.bool)]))

        return inputs, target

    def __len__(self):
        return len(self.patch_slices)


class CroppingDown2DDataset(Cropping2DDataset):
    def __init__(
            self,
            data, labels, rois, patch_size=32, overlap=16, filtered=False,
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
        downrois = [
            torch.max_pool2d(
                torch.tensor(np.expand_dims(roi, 0)).type(torch.float32), ratio
            ).squeeze(dim=0).numpy().astype(np.bool)
            for roi in rois
        ]
        super().__init__(downdata, downlabels, downrois, patch_size, overlap, filtered)
