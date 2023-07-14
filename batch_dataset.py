import numpy as np
import os.path as path
import random
import pickle
import io
import pandas as pd

from pathlib import Path
from PIL import Image
from astroquery.vizier import Vizier
from astropy import units as u

from utils import create_path, load_config


def match_mb(ra, dec, maj):
    match = 0

    rad = 0.5 * maj

    sky = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")

    while True:
        err = False
        try:
            result = Vizier.query_region(sky, radius=rad * u.arcsec, catalog=["J/MNRAS/466/4346/table1"])
        except requests.exceptions.ConnectionError as e:
            print("CONNECTION ERROR: {}".format(e))
            err = True

        if not err:
            break

    if len(result) > 0:
        match = 1

    return match


def make_csvfile(path, minsize, maxsize):
    filename = "static_rgz_flat_2019-05-06_full.csv"
    metadata = pd.read_csv(filename)

    n = 0
    df = None

    fits_path = path / "FITS"
    img_path = path / "PNG"

    # Get all img and fits files
    fits_list = [fits_path.stem for fits_path in fits_path.glob("*.fits")]
    img_list = [img_path.stem for img_path in img_path.glob("*.png")]

    # Make sure both img and fits exist for both
    id_list = list(set(fits_list).intersection(set(img_list)))

    # Drop unused columns and rename
    metadata = metadata[["rgz_name", "radio.ra", "radio.dec", "radio.max_angular_extent"]]
    metadata = metadata.rename(
        columns={
            "rgz_name": "source_id",
            "radio.ra": "ra",
            "radio.dec": "dec",
            "radio.max_angular_extent": "LAS",
        }
    )

    # Remove dataframe entries that don't have a match in id_list
    metadata = metadata[metadata["source_id"].isin(id_list)]

    # Filter based on size
    metadata = metadata[metadata["LAS"] > minsize]
    metadata = metadata[metadata["LAS"] < maxsize]

    # Add MiraBest column by applying match_mb to each row
    metadata["MiraBest"] = metadata.apply(
        lambda row: match_mb(row["ra"], row["dec"], row["LAS"]), axis=1
    )

    # Remove duplictes using source ID
    metadata = metadata[~metadata.duplicated(["Source ID"])]

    print(f"Data points: {len(metadata.index)} NaNs: {metadata.isnull().sum().sum()}")

    print("\n Writing to .csv file...")
    metadata.to_csv(f"RGZDR1Images.csv")

    return metadata


def filename_to_img(filename):
    img = Image.open(filename)
    img = np.array(img)
    img = np.array(list(img), np.uint8)

    return img


def build_dataset(path, metadata, n_batches=7):
    batch_dir = path / "batches"
    create_path(batch_dir)
    png_dir = path / "PNG"

    # All filenames in png directory
    filenames = sorted(list(png_dir.glob("*.png")))

    # Filter out classes 40x (unclassifiable) and 103 (diffuse FRI)
    filenames = [filename for filename in filenames if filename_to_label(filename) is not None]

    # Total number of pixels in each image
    nvis = np.prod(filename_to_img(filenames[0]).shape)

    print(f"Number of images (post filtering):{len(filenames)} \n")

    with open("test_names.pkl", "rb") as f:
        test_names = pickle.load(f)

    batches = []

    # Test batch
    batch = {
        "data": [],
        "filenames": [],
        "batch_label": "testing batch 1 of 1",
        "src_ids": [],
        "mb_flag": [],
        "LAS": [],
        "ra": [],
        "dec": [],
    }

    for test_name in test_names:
        test_name = list(png_dir.glob(f"{test_name[-38:-17]}*.png"))
        assert len(test_name) == 1
        test_name = test_name[0]
        batch["data"].append(filename_to_img(test_name))
        batch["filenames"].append(str(test_name))
        batch["src_ids"].append(test_name.stem)
        filenames.remove(test_name)

    with io.open(batch_dir / "test_batch", "wb") as f:
        pickle.dump(batch, f)

    print(f"Test batch containing {len(batch['labels'])} images saved\n")
    print(f"Train images remaining: {len(filenames)}\n")

    # Training batches
    n_batches = 7
    batch_size = len(filenames) // (n_batches - 1)

    print(f"Using a batch size of {batch_size} for a total of {n_batches} batches\n")

    # Seed batching
    random.seed(42)
    for i in range(n_batches):
        batch = {}
        batch = {
            "labels": [],
            "data": [],
            "filenames": [],
            "batch_label": f"training batch {i+1} of {n_batches}",
        }

        for j in range(min(batch_size, len(filenames))):
            filename = random.choice(filenames)
            batch["labels"].append(filename_to_label(filename))
            batch["data"].append(filename_to_img(filename))
            batch["filenames"].append(str(filename))
            filenames.remove(filename)

        with io.open(batch_dir / f"data_batch_{i+1}", "wb") as f:
            pickle.dump(batch, f)

        print(f"Batch {i+1} containing {len(batch['labels'])} images saved")

    # create dictionary of batch:
    metadata = {
        "num_cases_per_batch": batch_size,
        "label_names": ["100", "102", "104", "110", "112", "200", "201", "210", "300", "310"],
        "num_vis": nvis,
    }

    print(f"\nMetadata:")
    for key, value in metadata.items():
        print(f"{key}: {value}")

    with io.open(batch_dir / "batches.meta", "wb") as f:
        pickle.dump(dict, f)


if __name__ == "__main__":
    config = load_config()
    path = Path("MiraBest") / config["survey"]

    metadata = make_csvfile(path, minsize=config["minsize"], maxsize=config["maxsize"])

    build_dataset(metadata, path, n_batches=config["n_batches"])
