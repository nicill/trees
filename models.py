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
     [batch_size, data_shape]
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


def positive_uncertainty_loss(
        pred, target, q, q_factor=0.5, base=F.binary_cross_entropy
):
    """
    Flip loss function for the positive labels based on:
    Richard McKinley, Michael Rebsamen, Raphael Meier, Mauricio Reyes,
    Christian Rummel and Roland Wiest. "Few-shot brain segmentation from weakly
    labeled data with deep heteroscedastic multi-task network".
    https://arxiv.org/abs/1904.02436
    The idea is to allow for mislabeling inside the labeled are (since the area
    of the annotation is arbitrary, but the location of it is right). Otherwise
    the background label is correct.
    :param pred: Predicted values. The shape of the tensor should be related
     to the base function.
    :param target: Ground truth values. The shape of the tensor should be
     related to the base function.
    :param q: Uncertainty output from the network. The shape of the tensor
     should be related to the base function.
    :param q_factor: Factor to normalise the value of q.
    :param base: Base function for the flip loss.
    :return: The flip loss given a base loss function
    """
    norm_q = q * q_factor
    z = (pred < 0.5).type_as(pred) * target
    q_target = (1 - target) * norm_q + target * (1 - norm_q)
    loss_seg = base(pred, q_target.type_as(pred).detach())
    loss_uncertainty = base(norm_q, z.detach())

    final_loss = loss_seg + loss_uncertainty

    return final_loss


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
            n_inputs=4, n_outputs=1
    ):
        super(Unet2D, self).__init__()
        # Init values
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.device = device

        # <Parameter setup>
        self.autoencoder = Autoencoder2D(
            conv_filters, device, n_inputs, pooling=True
        )

        self.seg = nn.Sequential(
            nn.Conv2d(conv_filters[0], conv_filters[0], 1),
            nn.ReLU(),
            nn.BatchNorm2d(conv_filters[0]),
            nn.Conv2d(conv_filters[0], n_outputs, 1)
        )
        self.seg.to(device)

        self.unc = nn.Sequential(
            nn.Conv2d(conv_filters[0], conv_filters[0], 1),
            nn.ReLU(),
            nn.BatchNorm2d(conv_filters[0]),
            nn.Conv2d(conv_filters[0], n_outputs, 1)
        )
        self.unc.to(device)

        # <Loss function setup>
        self.train_functions = [
            {
                'name': 'xtop',
                'weight': 1,
                'f': lambda p, t: F.binary_cross_entropy(
                    p[0], torch.squeeze(t, dim=1).type_as(p[0]).to(p[0].device)
                )
            },
            {
                'name': 'dtop',
                'weight': 0.5,
                'f': lambda p, t: dsc_loss(p[0], t)
            },
            {
                'name': 'dbck',
                'weight': 0.5,
                'f': lambda p, t: dsc_loss(1 - p[0], t == 0)
            },
            {
                'name': 'unc',
                'weight': 1,
                'f': lambda p, t: positive_uncertainty_loss(
                    p[0],
                    torch.squeeze(t, dim=1).type_as(p[0]).to(p[0].device),
                    torch.squeeze(p[1], dim=1)
                )
            },
        ]
        self.val_functions = [
            {
                'name': 'xentr',
                'weight': 1,
                'f': lambda p, t: F.binary_cross_entropy(
                    p[0], torch.squeeze(t, dim=1).type_as(p[0]).to(p[0].device)
                )
            },
            {
                'name': 'dsc',
                'weight': 1,
                'f': lambda p, t: dsc_loss(p[0], t)
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
        multi_seg = torch.sigmoid(self.seg(input_s))

        unc = torch.sigmoid(self.seg(input_s))

        return multi_seg, unc

    def dropout_update(self):
        super().dropout_update()
        self.autoencoder.dropout = self.dropout

    def test(
            self, data, verbose=True
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

            seg_i = np.zeros(im.shape[1:])
            unc_i = np.zeros(im.shape[1:])

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
                    seg_pi, unc_pi = self(data_tensor)
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

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

            if verbose:
                print(
                    '\033[K%sSegmentation finished' % ' '.join([''] * 12)
                )

            seg.append(seg_i)
            unc.append(unc_i)
        return seg, unc
