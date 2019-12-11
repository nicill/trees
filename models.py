import torch
import torch.nn.functional as F
from data_manipulation.models import Autoencoder, BaseModel


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
        self.autoencoder = Autoencoder(
            conv_filters, device, n_images * 2,
        )

        self.seg = nn.Sequential(
            nn.Conv2d(conv_filters[0], conv_filters[0], 1),
            nn.ReLU(),
            nn.BatchNorm2d(conv_filters[0]),
            nn.Conv2d(conv_filters[0], 2, 1)
        )
        self.seg.to(device)

        # <Loss function setup>
        self.train_functions = [
            {
                'name': 'xentr',
                'weight': 1,
                'f': lambda p, t: F.binary_cross_entropy(
                    p[:, 1, ...],
                    torch.squeeze(t, dim=1).type_as(p).to(p.device)
                )
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
        ]

        # <Optimizer setup>
        # We do this last setp after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adadelta(model_params)
        # self.optimizer_alg = torch.optim.Adam(model_params, lr=1e-1)
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

        data_tensor = to_torch_var(data)

        with torch.no_grad():
            torch.cuda.synchronize()
            seg = self(data_tensor)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if verbose:
            print(
                '\033[K%sSegmentation finished' % ' '.join([''] * 12)
            )

        seg = list(map(np.squeeze, seg.cpu().numpy()))
        return seg