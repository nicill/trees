import itertools
import time
import numpy as np
import torch
from functools import partial
from torch import nn
import torch.nn.functional as F
from base import BaseModel
from utils import to_torch_var, time_to_string
from criteria import flip_loss, focal_loss, dsc_loss


class Autoencoder2D(BaseModel):
    def __init__(
            self,
            conv_filters,
            device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            ),
            n_inputs=1,
            kernel_size=3,
            pooling=False,
            dropout=0,
    ):
        super().__init__()
        # Init
        self.pooling = pooling
        self.device = device
        self.dropout = dropout
        # Down path
        self.down = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    f_in, f_out, kernel_size,
                    padding=kernel_size // 2,
                ),
                nn.ReLU(),
                nn.BatchNorm2d(f_out)
            ) for f_in, f_out in zip(
                [n_inputs] + conv_filters[:-2], conv_filters[:-1]
            )
        ])

        self.u = nn.Sequential(
            nn.Conv2d(
                conv_filters[-2], conv_filters[-1], kernel_size,
                padding=kernel_size // 2
            ),
            nn.ReLU(),
            nn.BatchNorm2d(conv_filters[-1])
        )

        # Up path
        down_out = conv_filters[-2::-1]
        up_out = conv_filters[:0:-1]
        deconv_in = map(sum, zip(down_out, up_out))
        self.up = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(
                    f_in, f_out, kernel_size,
                    padding=kernel_size // 2
                ),
                nn.ReLU(),
                nn.BatchNorm2d(f_out)
            ) for f_in, f_out in zip(
                deconv_in, down_out
            )
        ])

    def forward(self, input_s):
        down_inputs = []
        for c in self.down:
            c.to(self.device)
            input_s = F.dropout2d(
                c(input_s), self.dropout, self.training
            )
            down_inputs.append(input_s)
            if self.pooling:
                input_s = F.max_pool2d(input_s, 2)

        self.u.to(self.device)
        input_s = F.dropout2d(self.u(input_s), self.dropout, self.training)

        for d, i in zip(self.up, down_inputs[::-1]):
            d.to(self.device)
            if self.pooling:
                input_s = F.dropout2d(
                    d(
                        torch.cat(
                            (F.interpolate(input_s, size=i.size()[2:]), i),
                            dim=1
                        )
                    ),
                    self.dropout,
                    self.training
                )
            else:
                input_s = F.dropout2d(
                    d(torch.cat((input_s, i), dim=1)),
                    self.dropout,
                    self.training
                )

        return input_s


class Unet2D(BaseModel):
    def __init__(
            self,
            conv_filters=None,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            n_inputs=4, n_outputs=1
    ):
        super(Unet2D, self).__init__()
        # Init values
        if conv_filters is None:
            conv_filters = [32, 64, 128, 256]
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.device = device

        # <Parameter setup>
        # I plan to change that to use the AutoEncoder framework in
        # data_manipulation.
        self.autoencoder = Autoencoder2D(
            conv_filters, device, n_inputs, pooling=True
        )

        # Deep supervision branch.
        # This branch adapts the bottleneck filters to work with the final
        # segmentation block.
        self.deep_seg = nn.Sequential(
            nn.Conv2d(conv_filters[-1], conv_filters[0], 1),
            nn.ReLU(),
            nn.BatchNorm2d(conv_filters[0]),
        )
        self.deep_seg.to(device)

        # Final segmentation block.
        self.seg = nn.Sequential(
            nn.Conv2d(conv_filters[0], conv_filters[0], 1),
            nn.ReLU(),
            nn.BatchNorm2d(conv_filters[0]),
            nn.Conv2d(conv_filters[0], n_outputs, 1)
        )
        self.seg.to(device)

        # Final uncertainty block.
        # For now, this block is only used on the main branch. I am not sure
        # on how useful it might be for the deep branch.
        self.unc = nn.Sequential(
            nn.Conv2d(conv_filters[0], conv_filters[0], 1),
            nn.ReLU(),
            nn.BatchNorm2d(conv_filters[0]),
            nn.Conv2d(conv_filters[0], n_outputs, 1)
        )
        self.unc.to(device)

        # <Loss function setup>
        self.train_functions = [
            # Focal loss for the main branch.
            {
                'name': 'xentr',
                'weight': 1,
                'f': lambda p, t: focal_loss(
                    torch.squeeze(p[0], dim=1),
                    torch.squeeze(t, dim=1).type_as(p[0]).to(p[0].device),
                    alpha=0.5
                )
            },
            # Focal loss for the deep supervision branch (bottleneck).
            {
                'name': 'dp xe',
                'weight': 1,
                'f': lambda p, t: focal_loss(
                    torch.squeeze(p[2], dim=1),
                    torch.squeeze(
                        F.max_pool2d(
                            t.type_as(p[2]),
                            2 ** len(self.autoencoder.down)),
                        dim=1
                    ).to(p[2].device),
                    alpha=0.5
                )
            },
            # DSC loss for the main branch.
            {
                'name': 'dsc',
                'weight': 1,
                'f': lambda p, t: dsc_loss(p[0], t)
            },
            # DSC loss for the deep supervision branch (bottleneck).
            {
                'name': 'dp dsc',
                'weight': 1,
                'f': lambda p, t: dsc_loss(
                    p[2],
                    F.max_pool2d(
                        t.type_as(p[2]),
                        2 ** len(self.autoencoder.down)
                    ).to(p[2].device)
                )
            },
            # Uncertainty loss based on the flip loss (by Mckinley et al).
            {
                'name': 'unc',
                'weight': 1,
                'f': lambda p, t: flip_loss(
                    torch.squeeze(p[0], dim=1),
                    torch.squeeze(t, dim=1).type_as(p[0]).to(p[0].device),
                    torch.squeeze(p[1], dim=1),
                    q_factor=1,
                    base=partial(focal_loss, alpha=0.5)
                )
            },
        ]
        self.val_functions = [
            # Focal loss for validation.
            # The weight is 0 because I feel like DSC is more important.
            {
                'name': 'xentr',
                'weight': 0,
                'f': lambda p, t: focal_loss(
                    torch.squeeze(p[0], dim=1),
                    torch.squeeze(t, dim=1).type_as(p[0]).to(p[0].device),
                    alpha=0.5
                )
            },
            # DSC loss for validation.
            {
                'name': 'dsc',
                'weight': 1,
                'f': lambda p, t: dsc_loss(p[0], t)
            },
            # Losses based on uncertainty values.for
            # Their weight is 0 because I don't want them to affect early
            # stopping. They are useful values to see how the uncertainty is
            # evolving, but that's it.
            {
                'name': 'μ unc',
                'weight': 0,
                'f': lambda p, t: torch.mean(p[1])
            },
            {
                'name': 'unc',
                'weight': 0,
                'f': lambda p, t: torch.min(p[1])
            }
        ]
        self.acc_functions = [
            # Max uncertainty.
            # I use it as an accuracy function, because we want to maximise
            # them (losses are minimised) and I actually think that the
            # maximum uncertainty should rise and the mininum fall for it
            # to be any useful.
            # Since accuracy metrics are there just for added information
            # and they should not affect early stopping, they do not have
            # weights.
            {
                'name': 'UNC',
                'f': lambda p, t: torch.max(p[1])
            },
        ]

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        # self.optimizer_alg = torch.optim.Adadelta(model_params)
        self.optimizer_alg = torch.optim.Adam(model_params)
        # self.optimizer_alg = torch.optim.SGD(model_params, lr=1e-1)
        # self.autoencoder.dropout = 0.99
        # self.dropout = 0.99
        # self.ann_rate = 1e-2

    def forward(self, input_ae):
        input_s = self.autoencoder(input_ae)

        # Deep supervision.
        # We need to complete the down path and the bottleneck again.
        for c in self.autoencoder.down:
            input_ae = F.dropout2d(
                c(input_ae),
                self.autoencoder.dropout,
                self.autoencoder.training
            )
            input_ae = F.max_pool2d(input_ae, 2)
        input_ae = F.dropout2d(
            self.autoencoder.u(input_ae),
            self.autoencoder.dropout,
            self.autoencoder.training
        )

        # This is the last part of deep supervision
        input_ae = self.deep_seg(input_ae)

        # Since we are dealing with a binary problem, there is no need to use
        # softmax.
        multi_seg = torch.sigmoid(self.seg(input_s))
        unc = torch.sigmoid(self.unc(input_s))
        low_seg = torch.sigmoid(self.seg(input_ae))
        return multi_seg, unc, low_seg

    def dropout_update(self):
        super().dropout_update()
        self.autoencoder.dropout = self.dropout

    def test(
            self, data, patch_size=256, verbose=True
    ):
        # Init
        self.eval()
        seg = list()
        unc = list()

        # Init
        t_in = time.time()

        for i, im in enumerate(data):

            # Case init
            t_case_in = time.time()

            # This branch is only used when images are too big. In this case
            # they are split in patches and each patch is trained separately.
            # Currently, the image is partitioned in blocks with no overlap,
            # however, it might be a good idea to sample all possible patches,
            # test them, and average the results. I know both approaches
            # produce unwanted artifacts, so I don't know.
            if patch_size is not None:

                # Initial results. Filled to 0.
                seg_i = np.zeros(im.shape[1:])
                unc_i = np.zeros(im.shape[1:])

                limits = tuple(
                    list(range(0, lim, patch_size))[:-1] + [lim - patch_size]
                    for lim in data.shape[1:]
                )
                limits_product = list(itertools.product(*limits))

                n_patches = len(limits_product)

                # The following code is just a normal test loop with all the
                # previously computed patches.
                for pi, (xi, xj) in enumerate(limits_product):
                    # Here we just take the current patch defined by its slice
                    # in the x and y axes. Then we convert it into a torch
                    # tensor for testing.
                    xslice = slice(xi, xi + patch_size)
                    yslice = slice(xj, xj + patch_size)
                    data_tensor = to_torch_var(
                        np.expand_dims(im[slice(None), xslice, yslice], axis=0)
                    )

                    # Testing itself.
                    with torch.no_grad():
                        torch.cuda.synchronize()
                        seg_pi, unc_pi, _ = self(data_tensor)
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()

                    # Then we just fill the results image.
                    seg_i[xslice, yslice] = np.squeeze(seg_pi.cpu().numpy())
                    unc_i[xslice, yslice] = np.squeeze(unc_pi.cpu().numpy())

                    # Printing
                    init_c = '\033[0m' if self.training else '\033[38;5;238m'
                    whites = ' '.join([''] * 12)
                    percent = 20 * (pi + 1) // n_patches
                    progress_s = ''.join(['-'] * percent)
                    remainder_s = ''.join([' '] * (20 - percent))

                    t_out = time.time() - t_in
                    t_case_out = time.time() - t_case_in
                    time_s = time_to_string(t_out)

                    t_eta = (t_case_out / (pi + 1)) * (n_patches - (pi + 1))
                    eta_s = time_to_string(t_eta)
                    batch_s = '{:}Case {:03}/{:03} ({:03d}/{:03d})' \
                              ' [{:}>{:}] {:} ETA: {:}'.format(
                        init_c + whites, i + 1, len(data), pi + 1, n_patches,
                        progress_s, remainder_s, time_s, eta_s + '\033[0m'
                    )
                    print('\033[K', end='', flush=True)
                    print(batch_s, end='\r', flush=True)

            else:
                # If we use the whole image the process is way simpler.
                # We only need to convert the data into a torch tensor,
                # test it and return the results.
                data_tensor = to_torch_var(np.expand_dims(im, axis=0))

                # Testing
                with torch.no_grad():
                    torch.cuda.synchronize(self.device)
                    seg_pi, unc_pi, _ = self(data_tensor)
                    torch.cuda.synchronize(self.device)
                    torch.cuda.empty_cache()

                # Image squeezing.
                # The images have a batch number at the beginning. Since each
                # batch is just an image, that batch number is useless.
                seg_i = np.squeeze(seg_pi.cpu().numpy())
                unc_i = np.squeeze(unc_pi.cpu().numpy())

                # Printing
                init_c = '\033[0m' if self.training else '\033[38;5;238m'
                whites = ' '.join([''] * 12)
                percent = 20 * (i + 1)
                progress_s = ''.join(['-'] * percent)
                remainder_s = ''.join([' '] * (20 - percent))

                t_out = time.time() - t_in
                t_case_out = time.time() - t_case_in
                time_s = time_to_string(t_out)

                t_eta = (t_case_out / (i + 1)) * (len(data) - i + 1)
                eta_s = time_to_string(t_eta)
                batch_s = '{:}Case {:03}/{:03} [{:}>{:}] {:} ETA: {:}'.format(
                    init_c + whites, i + 1, len(data),
                    progress_s, remainder_s, time_s, eta_s + '\033[0m'
                )
                print('\033[K', end='', flush=True)
                print(batch_s, end='\r', flush=True)

            if verbose:
                print(
                    '\033[K%sSegmentation finished' % ' '.join([''] * 12)
                )

            seg.append(seg_i)
            unc.append(unc_i)

        return seg, unc
