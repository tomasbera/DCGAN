import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, nc, ngf, num_layers):
        super(Discriminator, self).__init__()
        self.ngf = ngf
        self.nc = nc

        # List to hold the layers
        layers = []

        # Dynamically create convolutional layers based on the number of layers
        for i in range(num_layers):
            in_channels = nc if i == 0 else ngf * (2 ** (i - 1))
            out_channels = ngf * (2 ** i)

            # Convolutional layer
            kernel_size = 3 if i == 0 else 4  # Reduced kernel size for the first layer
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, 2, 1, bias=False))

            # Batch normalization for intermediate layers
            if i != 0 and i != num_layers - 1:
                layers.append(nn.BatchNorm2d(out_channels))

            # Leaky ReLU activation
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        # AdaptiveAvgPool2d layer to ensure output size is (1, 1)
        layers.append(nn.AdaptiveAvgPool2d(1))

        # Output layer
        layers.append(nn.Conv2d(ngf * (2 ** (num_layers - 1)), 1, 1, 1, 0, bias=False))
        layers.append(nn.Sigmoid())

        # Sequential container to hold all layers
        self.main = nn.Sequential(*layers)

    def forward(self, data_in):
        intermediate_output = self.main(data_in)
        return intermediate_output
