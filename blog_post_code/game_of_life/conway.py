#!/usr/bin/python

import itertools as it
from os import PathLike
from typing import Generator, List
from pathlib import Path
from pkg_resources import resource_filename

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from numpy.lib import stride_tricks
from numpy.random import default_rng
from scipy import ndimage

from numpy.typing import NDArray

PLAINTEXT_GLIDER = (
    "......................O.O...........\n"
    "........................O...........\n"
    "............OO......OO............OO\n"
    "...........O...O....OO............OO\n"
    "OO........O.....O...OO..............\n"
    "OO........O...O.OO....O.O...........\n"
    "..........O.....O.......O...........\n"
    "...........O...O....................\n"
    "............OO......................"
)

IMAGE_DIR = Path(resource_filename("blog_post_code", "game_of_life/images"))

 
def create_random_grid(length: int = 10, width: int = 10, seed=None) -> NDArray:
    """Create a random 2D binary array sampled from a uniform distribution."""
    rng = default_rng(seed=seed)
    return rng.integers(low=0, high=2, size=length * width).reshape(length, width)


def count_neighbors_np(grid: NDArray) -> NDArray:
    """Count neighbors using numpy's sliding_window_view, treating off-grid neighbors as dead.
    Returns an array of the same size."""

    padded = np.pad(grid, 1)
    windows = stride_tricks.sliding_window_view(padded, (3, 3))

    return windows.sum(axis=(2, 3)) - grid


def count_neighbors(grid: NDArray) -> NDArray:
    """Count the live neighbors of each cell in a 2D binary array
    treating off-grid neighbors as dead. Returns an np.array of
    the same size as the input."""
    kernel = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1]).reshape(3, 3)
    return ndimage.convolve(grid, kernel, mode="constant", cval=0.0)


def apply_conways_rules(grid: NDArray) -> NDArray:
    """
    Step forward one generation according to the following rules:

    1) Any live cell with two or three live neighbours survives.
    2) Any dead cell with three live neighbours becomes a live cell.
    3) All other live cells die in the next generation. Similarly, all other dead cells stay dead.
    """

    neighbors = count_neighbors(grid)
    return ((neighbors == 2) & grid) | (neighbors == 3)


def simulate(grid: NDArray) -> Generator[NDArray, None, None]:
    """Simulate infinite generations of the Game of Life provided a starting grid."""

    while True:
        grid = apply_conways_rules(grid)
        yield grid


def convert_plaintext(text: str) -> NDArray:
    """Provided a plaintext game of life code where each row has the same number of elements,
    return the corresponding numpy array."""

    replaced_digits: List[str] = text.replace(".", "0").replace("O", "1").split()
    exploded: List[List[str]] = list(map(list, replaced_digits))

    return np.array(exploded).astype("int")


def save_conway_to_gif(grid: NDArray, filepath: PathLike[str], nframes: int = 100) -> None:
    """
    Create a gif of the first nframes of
    evolution of the provided grid and save it to filepath.
    """

    # create a list of arrays representing the first NFRAMES generations
    data = list(it.islice(simulate(grid), 0, nframes))

    # create an initial matrix figure without ticks or labels
    fig = plt.figure()
    plot = plt.matshow(data[0], fignum=0, cmap=plt.get_cmap("binary")) # type: ignore
    plt.tick_params(
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False,
        bottom=False,
        top=False,
        labeltop=False,
    )

    def init():
        """
        Provides the initial plot
        (to be passed to FuncAnimation's init_func arg as a callable)
        """
        plot.set_data(data[0])
        return [plot]

    def update(j):
        """
        Updates the figure's data in place
        (to be passed to the FunAnimation func arg as a callable).
        """
        plot.set_data(data[j])
        return [plot]

    metadata = dict(title="Conway's Game of Life", artist="Will", comment="")
    writer = animation.FFMpegWriter(fps=5, metadata=metadata, bitrate=3500)
    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=nframes, interval=30, blit=True
    )
    anim.save(str(filepath), writer=writer, dpi=300)


def main() -> None:
    """Create 2 gifs: one with a random initial grid, and one with a glider gun."""

    random_grid = create_random_grid(34, 61)
    glider_grid = np.pad(convert_plaintext(PLAINTEXT_GLIDER), (10, 15))

    save_conway_to_gif(random_grid, IMAGE_DIR / "random1.gif")
    save_conway_to_gif(glider_grid, IMAGE_DIR / "glider1.gif", nframes=90)


if __name__ == "__main__":
    main()  # pragma: no cover
