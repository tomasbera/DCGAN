import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, nc, ngf):
        super(Discriminator, self).__init__()
        self.ngf = ngf
        self.nc = nc

        # Initialize the neural network with specified input channels and filter count.
        #   Args:
        #     - nc (int): Number of input channels.
        #     - ngf (int): Number of filters in the first convolutional layer.
        self.main = nn.Sequential(
            # Layer 1
            nn.Conv2d(nc, ngf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Output Layer
            nn.Conv2d(ngf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, data_in):
        return self.main(data_in)
