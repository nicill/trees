import itertools
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from data_manipulation.models import BaseModel
from data_manipulation.utils import to_torch_var, time_to_string


def dsc_loss(pred, target, smooth=0.1):
    """
    Loss function based on a single class DSC metric.
    :param pred: Predicted values. This tensor should have the shape:
     [batch_size, n_classes, data_shape]
    :param target: Ground truth values. This tensor can have multiple shapes:
     - [batch_size, n_classes, data_shape]: This is the expected output since
       it matches with the predicted tensor.
     - [batch_size, data_shape]: In this case, the tensor is labeled with
       values ranging from 0 to n_classes. We need to convert it to
       categorical.
    :param smooth: Parameter used to smooth the DSC when there are no positive
     samples.
    :return: The mean DSC for the batch
    """
    dims = pred.shape
    assert target.shape == pred.shape,\
        'Sizes between predicted and target do not match'
    target = target.type_as(pred)

    reduce_dims = tuple(range(1, len(dims)))
    num = (2 * torch.sum(pred * target, dim=reduce_dims))
    den = torch.sum(pred + target, dim=reduce_dims) + smooth
    dsc_k = num / den
    dsc = 1 - torch.mean(dsc_k)

    return torch.clamp(dsc, 0., 1.)


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
            conv_filters=list([32, 64, 128, 256]),
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            n_inputs=4, n_outputs=2
    ):
        super(Unet2D, self).__init__()
        # Init values
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.device = device

        # <Parameter setup>
        self.autoencoder = Autoencoder2D(
            conv_filters, device, n_inputs,
        )

        self.seg = nn.Sequential(
            nn.Conv2d(conv_filters[0], conv_filters[0], 1),
            nn.ReLU(),
            nn.BatchNorm2d(conv_filters[0]),
            nn.Conv2d(conv_filters[0], n_outputs, 1)
        )
        self.seg.to(device)

        # <Loss function setup>
        self.train_functions = [
            {
                'name': 'xetop',
                'weight': 1,
                'f': lambda p, t: F.binary_cross_entropy(
                    p[:, 1, ...],
                    torch.squeeze(t, dim=1).type_as(p).to(p.device)
                )
            },
            {
                'name': 'dtop',
                'weight': 0.5,
                'f': lambda p, t: dsc_loss(
                    torch.unsqueeze(p[:, 1, ...], 1), t)
            },
            {
                'name': 'dbck',
                'weight': 0.5,
                'f': lambda p, t: dsc_loss(
                    torch.unsqueeze(p[:, 0, ...], 1), t == 0)
            },
        ]
        self.val_functions = [
            {
                'name': 'xentr',
                'weight': 1,
                'f': lambda p, t: F.binary_cross_entropy(
                    p[:, 1, ...],
                    torch.squeeze(t, dim=1).type_as(p).to(p.device)
                )
            },
            {
                'name': 'dsc',
                'weight': 1,
                'f': lambda p, t: dsc_loss(
                    torch.unsqueeze(p[:, 1, ...], 1), t)
            },
        ]

        # <Optimizer setup>
        # We do this last setp after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        # self.optimizer_alg = torch.optim.Adadelta(model_params)
        self.optimizer_alg = torch.optim.Adam(model_params, lr=1e-2)
        # self.optimizer_alg = torch.optim.SGD(model_params, lr=1e-1)
        # self.autoencoder.dropout = 0.99
        # self.dropout = 0.99
        # self.ann_rate = 1e-2

    def forward(self, input_s):
        input_s = self.autoencoder(input_s)
        multi_seg = torch.softmax(self.seg(input_s), dim=1)

        return multi_seg

    def dropout_update(self):
        super().dropout_update()
        self.autoencoder.dropout = self.dropout

    def test(
            self, data, verbose=True
    ):
        # Init
        self.eval()
        seg = list()

        # Init
        t_in = time.time()

        for i, im in enumerate(data):

            # Case init
            t_case_in = time.time()

            seg_i = np.zeros(im.shape[1:])

            limits = tuple(
                list(range(0, lim, 256))[:-1] + [lim] for lim in im.shape[1:]
            )
            limits_product = list(itertools.product(
                range(len(limits[0]) - 1), range(len(limits[1]) - 1)
            ))
            n_patches = len(limits_product)
            for pi, (xi, xj) in enumerate(limits_product):
                xslice = slice(limits[0][xi], limits[0][xi + 1])
                yslice = slice(limits[1][xj], limits[1][xj + 1])
                data_tensor = to_torch_var(
                    np.expand_dims(im[slice(None), xslice, yslice], axis=0)
                )

                with torch.no_grad():
                    torch.cuda.synchronize()
                    out_i = np.squeeze(self(data_tensor).cpu().numpy())
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                seg_i[xslice, yslice] = out_i[1, ...]

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

            if verbose:
                print(
                    '\033[K%sSegmentation finished' % ' '.join([''] * 12)
                )

            seg.append(seg_i)
        return seg
