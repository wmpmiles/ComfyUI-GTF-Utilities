# import hack
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

# actual imports
from pathlib import Path
import argparse
import torch
import matplotlib.pyplot as plt
import resample


def normalize(tensor):
    tensor -= torch.min(tensor)
    tensor /= torch.max(tensor)
    return tensor

def plot(name, resampler, args, data, tos, path, size = (10, 10), cmap = "viridis"):
    shape = tuple(map(int, tuple(data.shape)))

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(*size)
    fig.tight_layout()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    def plot_ref():
        title = f"{name}: {shape} | {args}"
        filename = f"{name}_{args}_{shape}_ref.png"
        fig.suptitle(title)
        ref = data
        if ref.dim() == 1:
            ref = ref.unsqueeze(0)
        ax.imshow(ref, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        plt.savefig(path / filename)

    plot_ref()

    def plot(to):
        title = f"{name}: {shape} -> {to} | {args}"
        filename = f"{name}_{args}_{shape}_{to}.png"
        fig.suptitle(title)
        resampled = resampler(data, to)
        if resampled.dim() == 1:
            resampled = resampled.unsqueeze(0)
        ax.imshow(resampled, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        plt.savefig(path / filename)

    for to in tos:
        plot(to)
    
    plt.close()


def plots_1d(resampler, args, path):
    size = (20, 2)

    # random
    data = normalize(torch.randn(500))
    tos = (1000, 999, 750, 749, 499, 250, 249, 100)
    plot("rand", resampler, args, data, tos, path, size)

    # linear 
    data = torch.arange(500) / 499
    tos = (1000, 999, 499, 250, 100, 50, 10)
    plot("linear", resampler, args, data, tos, path, size)

    # checkerboard
    data = torch.cat((torch.zeros(100).reshape(1,-1), torch.ones(100).reshape(1, -1)), 0).transpose(0, 1).flatten()
    tos = (400, 399, 300, 299, 199, 190, 100, 99, 50, 49, 10, 9)
    plot("oscillating", resampler, args, data, tos, path, size)


def plots_2d(resampler, args, path):
    tos = ((200, 200), (199, 199), (150, 150), (99, 99), (75, 75), (50, 50), (49, 49), (10, 10), (9, 9))

    # random
    data = normalize(torch.randn(100, 100))
    plot("rand", resampler, args, data, tos, path)

    # linear
    data = (torch.arange(100).unsqueeze(1) + torch.arange(100)) / 198
    plot("linear", resampler, args, data, tos, path)

    # checkerboard
    data = torch.cat((torch.zeros(50), torch.ones(50)), 0).reshape(2,50).transpose(0,1).reshape(100,1)
    data = torch.cat((data, data.flip(0)),1).repeat(1,50)
    plot("checkerboard", resampler, args, data, tos, path)


def plot_select(resampler: str):
    PATH = Path("test_images")
    SEED = 0

    match resampler:
        case "nearest-neighbor":
            torch.manual_seed(SEED)
            args = ()
            resampler_1d = resample.nearest_neighbor_resample_1d
            resampler_2d = resample.nearest_neighbor_resample_2d
            path = PATH / "nearest_neighbor"
            os.makedirs(path, exist_ok=True)
            plots_1d(resampler_1d, args, path)
            plots_2d(resampler_2d, args, path)
        case "triangle":
            torch.manual_seed(SEED)
            radii = (1, 2, 3)
            filters = tuple(map(resample.triangle_filter, radii))
            path = PATH / "triangle"
            os.makedirs(path, exist_ok=True)
            for (radius, filter) in zip(radii, filters):
                resampler_1d = lambda d_s, L: resample.filter_resample_1d(d_s, L, radius, filter)
                resampler_2d = lambda d_s, L: resample.filter_resample_2d(d_s, L, radius, filter)
                args = (radius, )
                plots_1d(resampler_1d, args, path)
                plots_2d(resampler_2d, args, path)
                resampler_2d = lambda d_s, L: resample.filter_resample_2d_seperable(d_s, L, radius, filter)
                args = (radius, "seperable")
                plots_2d(resampler_2d, args, path)
        case "lanczos":
            torch.manual_seed(SEED)
            path = PATH / "lanczos"
            os.makedirs(path, exist_ok=True)
            for radius in (3, 4):
                filter = resample.lanczos_filter(radius)
                resampler_1d = lambda d_s, L: resample.filter_resample_1d(d_s, L, radius, filter)
                resampler_2d = lambda d_s, L: resample.filter_resample_2d(d_s, L, radius, filter)
                args = (radius, )
                plots_1d(resampler_1d, args, path)
                plots_2d(resampler_2d, args, path)
                resampler_2d = lambda d_s, L: resample.filter_resample_2d_seperable(d_s, L, radius, filter)
                args = (radius, "seperable")
                plots_2d(resampler_2d, args, path)
        case "mitchell-netravali":
            torch.manual_seed(SEED)
            path = PATH / "mitchell-netravali"
            os.makedirs(path, exist_ok=True)
            for (b, c) in ((0, 0), (1, 1), (0, 1), (1, 0), (0.33, 0.33)):
                filter = resample.mitchell_netravali_filter(b, c)
                resampler_1d = lambda d_s, L: resample.filter_resample_1d(d_s, L, 2, filter)
                resampler_2d = lambda d_s, L: resample.filter_resample_2d(d_s, L, 2, filter)
                args = (b, c)
                plots_1d(resampler_1d, args, path)
                plots_2d(resampler_2d, args, path)
                resampler_2d = lambda d_s, L: resample.filter_resample_2d_seperable(d_s, L, 2, filter)
                args = (b, c, "seperable")
                plots_2d(resampler_2d, args, path)
        case "area":
            torch.manual_seed(SEED)
            path = PATH / "area"
            os.makedirs(path, exist_ok=True)
            args = ()
            resampler_1d = resample.area_resample_1d
            resampler_2d = resample.area_resample_2d
            plots_1d(resampler_1d, args, path)
            plots_2d(resampler_2d, args, path)
        case _:
            raise ValueError(f"Unknown resampler: {resampler}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("resampler")
    args = parser.parse_args()
    plot_select(args.resampler)