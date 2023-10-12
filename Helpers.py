import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np


def print_images(dataloader, img_list, device):
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


def show_starting_img(dataloader, device, title="Training Images", grid_size=(8, 8)):
    # Get a batch of real images
    real_batch = next(iter(dataloader))

    # Ensure grid size is within the bounds of the batch size
    grid_size = (min(grid_size[0], real_batch[0].size(0)), grid_size[1])

    # Plot the images
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title(title)

    # Display a grid of images
    image_grid = vutils.make_grid(real_batch[0].to(device)[:grid_size[0] * grid_size[1]], padding=2, normalize=True)
    plt.imshow(np.transpose(image_grid.cpu(), (1, 2, 0)))
    plt.show()


def show_graph(G_losses, D_losses):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
