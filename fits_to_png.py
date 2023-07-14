import numpy as np

from PIL import Image
from pathlib import Path
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from utils import create_path, array_to_png, load_config


def read_fits_image(filepath):
    """
    This function reads in a fits file, preprocesses it and converts it to a png image.

    Args:
        filepath: Path to the fits file to read in
    """

    dir = filepath.parent / "PNG"
    create_path(dir)

    img = fits.getdata(filepath)

    # Remove nans
    img[np.where(np.isnan(img))] = 0.0

    # Sigma clipping
    _, _, rms = sigma_clipped_stats(img)
    img[np.where(img <= 3 * rms)] = 0.0

    # normalise to [0, 1]:
    image_max, image_min = img.max(), img.min()
    img = (img - image_min) / (image_max - image_min)

    # remap to [0, 255] for greyscale:
    img *= 255.0

    img = array_to_png(img)
    img.save(dir / (filepath.stem + ".png"))

    return img


if __name__ == "__main__":
    config = load_config()
    dir = Path("RGZ") / config["survey"] / "FITS"

    list = dir.glob("*.fits")

    for file in list:
        # Create images for any file not in the blacklist
        print(f"Saving png file: {str(file)}")
        read_fits_image(file)
