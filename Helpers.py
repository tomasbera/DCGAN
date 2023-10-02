import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np


def print_images(dataloader, img_list):
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(12, 12))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0))
    )

    # Plot the fake images fom next iter
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()
