from torch.utils.data.dataset import Dataset


class GenericSegmentationCroppingDataset(Dataset):
    def __init__(
            self,
            data, labels, patch_size=32
    ):
        # Init
        self.neg_ratio = neg_ratio
        # Image and mask should be numpy arrays
        self.sampler = sampler
        self.cases = cases
        self.labels = labels

        self.masks = masks

        data_shape = self.cases[0].shape

        if type(patch_size) is not tuple:
            patch_size = (patch_size,) * len(data_shape)
        self.patch_size = patch_size

        self.patch_slices = []
        if not self.sampler and balanced:
            if self.masks is not None:
                self.patch_slices = get_balanced_slices(
                    self.labels, self.patch_size, self.masks,
                    neg_ratio=self.neg_ratio
                )
            elif self.labels is not None:
                self.patch_slices = get_balanced_slices(
                    self.labels, self.patch_size, self.labels,
                    neg_ratio=self.neg_ratio
                )
            else:
                data_single = map(
                    lambda d: np.ones_like(
                        d[0] if len(d) > 1 else d
                    ),
                    self.cases
                )
                self.patch_slices = get_slices_bb(data_single, self.patch_size, 0)
        else:
            overlap = tuple(int(p // 1.1) for p in self.patch_size)
            if self.masks is not None:
                self.patch_slices = get_slices_bb(
                    self.masks, self.patch_size, overlap=overlap,
                    filtered=True
                )
            elif self.labels is not None:
                self.patch_slices = get_slices_bb(
                    self.labels, self.patch_size, overlap=overlap,
                    filtered=True
                )
            else:
                data_single = map(
                    lambda d: np.ones_like(
                        d[0] > np.min(d[0]) if len(d) > 1 else d
                    ),
                    self.cases
                )
                self.patch_slices = get_slices_bb(
                    data_single, self.patch_size, overlap=overlap,
                    filtered=True
                )
        self.max_slice = np.cumsum(list(map(len, self.patch_slices)))

    def __getitem__(self, index):
        # We select the case
        case_idx = np.min(np.where(self.max_slice > index))
        case = self.cases[case_idx]

        slices = [0] + self.max_slice.tolist()
        patch_idx = index - slices[case_idx]
        case_slices = self.patch_slices[case_idx]

        # We get the slice indexes
        none_slice = (slice(None, None),)
        slice_i = case_slices[patch_idx]

        inputs = case[none_slice + slice_i].astype(np.float32)

        if self.labels is not None:
            labels = self.labels[case_idx].astype(np.uint8)
            target = np.expand_dims(labels[slice_i], 0)

            if self.sampler:
                return inputs, target, index
            else:
                return inputs, target
        else:
            return inputs, case_idx, slice_i

    def __len__(self):
        return self.max_slice[-1]
