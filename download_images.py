import requests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy.io import fits
from astroquery.skyview import SkyView
from astropy.coordinates import SkyCoord
from astropy import units as u
from pathlib import Path
from urllib.request import urlretrieve
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from utils import create_path, load_config


def image_download(filename, ra, dec, survey="VLA FIRST (1.4 GHz)", pixels=150):
    """Download an image from an entry of the same format as previously"""

    print(f"Attempting to query SkyView for image... (RA: {ra}, DEC: {dec})")

    # Get co ordinates
    sky = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    url = SkyView.get_image_list(position=sky, survey=survey, cache=False, pixels=pixels)

    # Download .fits file from SkyView server
    try:
        file = requests.get(url[0], allow_redirects=True)
    except:
        print("Unable to download", filename)
        return None

    # Write .fits file to disk
    try:
        open(filename, "wb").write(file.content)
    except:
        print("No FITS available:", filename)
        return None

    # Plot image as sanity check
    hdu = fits.open(filename)
    img = np.squeeze(hdu[0].data)

    plt.imshow(img, cmap="hot")
    plt.savefig("fits_img.png")
    plt.close()


if __name__ == "__main__":
    config = load_config()

    # Download files from SkyView
    metadata = pd.read_csv("static_rgz_flat_2019-05-06_full.csv")

    dir = Path("RGZ") / config["survey"] / "FITS"
    create_path(dir)

    n_downloads = 0
    for i, row in tqdm(metadata.iterrows()):
        id = str(row["rgz_name"])
        size = float(row["radio.max_angular_extent"])

        path = dir / f"{id}.fits"

        if path.exists():
            print(f"FITS file {path} already exists - manually delete to re-download")
        elif size < config["min_size"]:
            print(f"Skipping {id} (size {size} too small)")
        elif size > config["max_size"]:
            print(f"Skipping {id} (size {size} too large)")
        else:
            image_download(
                path,
                row["radio.ra"],
                row["radio.dec"],
                survey=config["survey"],
                pixels=config["crop_size"],
            )

    print(f"Number of expected files: {n_downloads}")
    print(f"Number of downloaded files: {len(list(dir.glob('*.fits')))}")
