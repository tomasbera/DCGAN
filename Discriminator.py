import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ngf):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.ngf = ngf
        self.nc = nc

        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(self.nc, self.ngf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ngf, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ngf * 2, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ngf * 4, self.ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ngf * 8, 1, 4, 1, 0, bias=False),  # Corrected this line
            nn.Sigmoid()
        )

    def forward(self, data_in):
        return self.main(data_in)
