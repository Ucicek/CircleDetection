import numpy as np
from matplotlib import pyplot as plt
from skimage.draw import circle_perimeter_aa
from typing import NamedTuple, Optional, Tuple, Generator


class CircleParams(NamedTuple):
    row: int
    col: int
    radius: int


def draw_circle(img: np.ndarray, row: int, col: int, radius: int) -> np.ndarray:
    """
    Draw an anti-aliased white circle on a black background in a numpy array.

    Args:
        img (np.ndarray): A numpy array representing the image.
        row (int): The row index of the circle's center.
        col (int): The column index of the circle's center.
        radius (int): The radius of the circle.

    Returns:
        np.ndarray: The modified image array with the circle drawn on it.
    """

def noisy_circle(
        img_size: int, min_radius: float, max_radius: float, noise_level: float
) -> Tuple[np.ndarray, CircleParams]:
    """
    Draw a circle with random center and radius in a numpy array, and add normal noise.

    Args:
        img_size (int): Size of the square image (width = height = img_size).
        min_radius (float): Minimum radius of the circle.
        max_radius (float): Maximum radius of the circle.
        noise_level (float): Standard deviation of the Gaussian noise added to the image.

    Returns:
        Tuple[np.ndarray, CircleParams]: A tuple containing the noisy image and the 
                                         CircleParams object with circle details.
    """
    
    # Create an empty image
    img = np.zeros((img_size, img_size))

    radius = np.random.randint(min_radius, max_radius)

    # x,y coordinates of the center of the circle
    row, col = np.random.randint(img_size, size=2)

    # Draw the circle inplace
    draw_circle(img, row, col, radius)

    added_noise = np.random.normal(0.5, noise_level, img.shape)
    img += added_noise

    return img, CircleParams(row, col, radius)


def show_circle(img: np.ndarray):
    """
    Display a grayscale image containing a circle.

    Args:
        img (np.ndarray): A numpy array representing the image with a circle.

    No return value; this function only displays the image.
    """
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.set_title('Circle')
    plt.show()


def generate_examples(
        noise_level: float = 0.5,
        img_size: int = 100,
        min_radius: Optional[int] = None,
        max_radius: Optional[int] = None,
) -> Generator[Tuple[np.ndarray, CircleParams], None, None]:
    """
    Generator function that yields images of circles with random radius and position, 
    with added noise.

    Args:
        noise_level (float, optional): Standard deviation of the Gaussian noise added to the image. 
                                       Defaults to 0.5.
        img_size (int, optional): Size of the square image (width = height). Defaults to 100.
        min_radius (Optional[int], optional): Minimum radius of the circles. Defaults to img_size // 10.
        max_radius (Optional[int], optional): Maximum radius of the circles. Defaults to img_size // 2.

    Yields:
        Generator[Tuple[np.ndarray, CircleParams], None, None]:
    """
    if not min_radius:
        min_radius = img_size // 10
    if not max_radius:
        max_radius = img_size // 2
    assert max_radius > min_radius, "max_radius must be greater than min_radius"
    assert img_size > max_radius, "size should be greater than max_radius"
    assert noise_level >= 0, "noise should be non-negative"
    np.random.seed(42)

    while True:
        img, params = noisy_circle(
            img_size=img_size, min_radius=min_radius, max_radius=max_radius, noise_level=noise_level
        )
        yield img, params
