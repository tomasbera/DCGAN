import argparse
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

# Define Generator class (in Generator.py)
from Generator import Generator as Gen
# Define Discriminator class (in Discriminator.py)
from Discriminator import Discriminator as Dis
# Define DCGAN class (in DCGAN.py)
from Gan import Gan

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True)


def data_loader(workers, batch_size, image_size):
    ds_root = "./datasets"
    dataroot = f'{ds_root}/celeba-dataset'

    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0))
    )

    return dataloader, device


# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def run(num_workers, batch_size, num_epochs, lr):
    image_size = 64
    nc = 3
    nz = 100
    ngf = 64
    ndf = 64
    beta1 = 0.5

    # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        # Get the number of available GPUs
        gpu_count = torch.cuda.device_count()
        print(f"Number of available GPUs: {gpu_count}")
    else:
        print("CUDA is not available on this system.")

    # Set device
    dataloader, device = data_loader(num_workers, batch_size, image_size)
    gpu_count = torch.cuda.device_count()

    netG = Gen(gpu_count, nz, ngf, nc).to(device)
    netD = Dis(gpu_count, nc, ndf).to(device)
    gan = Gan(num_epochs, dataloader, netD, netG, device, nz, nc).to(device)

    if (device.type == 'cuda') and (gpu_count > 1):
        netG = nn.DataParallel(netG, list(range(gpu_count)))
        netD = nn.DataParallel(netD, list(range(gpu_count)))

    netG.apply(weights_init)
    netD.apply(weights_init)

    # Initialize the ``BCELoss`` function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    gan.train(real_label, fake_label, criterion, optimizerD, optimizerG, fixed_noise)


if __name__ == '__main__':
    run(2, 128, 5, 0.001)
