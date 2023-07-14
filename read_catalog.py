from astropy.io import fits
from astroquery.skyview import SkyView
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.vizier import Vizier
import argparse
import logging
import requests
import numpy as np
import pandas as pd
import os
import torch

from einops import rearrange
from torchvision.transforms.functional import center_crop, resize
from tqdm import tqdm
from PIL import Image
from astropy.stats import sigma_clipped_stats
from utils import create_path


parser = argparse.ArgumentParser(description="Parameters for batching")
# parser.add_argument("-c", "--clipping", type=bool, default=True, help="Apply clipping")
parser.add_argument("-d", "--dld", type=bool, default=True, help="Download fits")
parser.add_argument("-i", "--img", type=bool, default=True, help="Create img files")
parser.add_argument("-s", "--csv", type=bool, default=True, help="Create csv file")
parser.add_argument(
    "-su", "--survey", type=str, default="FIRST", help="Choose which survey to get data from"
)
args = parser.parse_args()


survey_strings = {"FIRST": "VLA FIRST (1.4 GHz)", "nvss": "NVSS"}


def rescale_image(img, low):
    img_max = np.max(img)
    img_min = low * 1e-3
    # img -= img_min
    img /= max(1e-6, img_max - img_min)  # clamp divisor so it can't be zero
    # img /= img_max - img_min
    img *= 255.0

    return img


def crop_centre(img, crop=150):
    xsize = np.shape(img)[0]  # image width
    ysize = np.shape(img)[1]  # image height
    startx = xsize // 2 - (crop // 2)
    starty = ysize // 2 - (crop // 2)
    sub_img = img[startx : startx + crop, starty : starty + crop]

    return sub_img


def apply_circular_mask(img, maj, frac=0.6):
    centre = (np.rint(img.shape[0] / 2), np.rint(img.shape[1] / 2))
    maj = frac * maj / 1.8  # arcsec --> pixels

    Y, X = np.ogrid[: img.shape[1], : img.shape[1]]
    dist_from_centre = np.sqrt((X - centre[0]) ** 2 + (Y - centre[1]) ** 2)

    mask = dist_from_centre <= maj

    img *= mask.astype(int)

    return img


def create_png(image_data, name="", path=""):
    im = Image.fromarray(image_data)
    im = im.convert("L")
    im.save(path + name + ".png")

    return


def get_fits(id, ra, dec, overwrite=True):
    create_path("fits", f"fits/{args.survey}")

    if not os.path.exists("fits"):
        os.mkdir("fits")

    fitsname = f"fits/{args.survey}/" + id + ".fits"
    if overwrite is True or not os.path.exists(fitsname):
        sky = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
        url = SkyView.get_image_list(
            position=sky, survey=survey_strings[args.survey], cache=False, pixels=150
        )
        try:
            file = requests.get(url[0], allow_redirects=True)
        except:
            print("No FITS available:", fitsname)
            return None

        try:
            open(fitsname, "wb").write(file.content)
        except:
            print("No FITS available:", fitsname)
            return None

    return fitsname


def plot_image(fitsfile, ra, dec, maj, low):
    if not os.path.exists("img"):
        os.mkdir("img")

    # get data:
    try:
        hdu = fits.open(fitsfile)
    except:
        print("Removing corrupt FITS: {}".format(fitsfile))
        os.remove(fitsfile)
        return None

    img = np.squeeze(hdu[0].data)
    # pl.subplot(161)
    # pl.imshow(data)
    # pl.title("Orig")

    # crop centre:
    img = crop_centre(img, crop=150)
    # pl.subplot(162)
    # pl.imshow(img)
    # pl.title("Crop")

    # radial crop:
    img = apply_circular_mask(img, maj)
    # pl.subplot(163)
    # pl.imshow(img)
    # pl.title("Mask")

    # Remove nans:
    img[np.where(np.isnan(img))] = 0.0

    # Clip background noise
    _, _, rms = sigma_clipped_stats(img)
    img[np.where(img <= 3 * rms)] = 0.0
    # pl.subplot(164)
    # pl.imshow(img)
    # pl.title("NaN")

    # subtract 3 sigma noise:
    img[np.where(img <= low * 1e-3)] = 0.0
    # pl.subplot(165)
    # pl.imshow(img)
    # pl.title("Sigma clip")

    # rescale image:
    img = rescale_image(img, low)
    # pl.subplot(166)
    # pl.imshow(img)
    # pl.title("Rescale")
    # pl.show()

    # create PNG:
    create_path("img", f"img/{args.survey}")

    pngname = f"./img/{args.survey}/" + ".".join(fitsfile.split(".")[0:2]).split("/")[1]
    if np.sum(img) > 0.0:
        create_png(img, name=pngname, path="./")
    else:
        print("Image is just zeros: {}".format(pngname))
        return None

    return pngname.split("/")[2] + ".png"


def update_csv(id, imgname, majaxis, match, df=None):
    info = {
        "Source ID": [id],
        "Map File": [imgname],
        "LAS": [majaxis],
        "MiraBest": [match],
    }

    df_tmp = pd.DataFrame(info, columns=["Source ID", "Map File", "LAS", "MiraBest"])

    if isinstance(df, pd.DataFrame):
        # df = df.append(df_tmp, ignore_index=True)
        df = pd.concat([df, df_tmp], ignore_index=True)
    else:
        df = df_tmp

    return df


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


def get_all_fits(minsize=None, maxsize=None):
    filename = "static_rgz_flat_2019-05-06_full.csv"
    data = pd.read_csv(filename)

    n = 0
    for i in tqdm(range(len(data))):
        item = data.iloc[i, :]
        majaxis = float(item["radio.max_angular_extent"])  # arcsec

        if minsize < majaxis < maxsize:
            id = str(item["rgz_name"])

            ra = float(item["radio.ra"])
            dec = float(item["radio.dec"])

            # get the FITS file:
            fitsfile = get_fits(id, ra, dec, overwrite=False)
            if fitsfile is not None:
                n += 1

    return n


def get_all_imgs(minsize, maxsize):
    filename = "static_rgz_flat_2019-05-06_full.csv"
    data = pd.read_csv(filename)

    n = 0
    for i in tqdm(range(len(data))):
        item = data.iloc[i, :]
        majaxis = float(item["radio.max_angular_extent"])  # arcsec

        if minsize < majaxis < maxsize:
            id = str(item["rgz_name"])

            ra = float(item["radio.ra"])
            dec = float(item["radio.dec"])

            # if args.clipping:
            #     low = float(item["radio.outermost_level"])
            # else:
            #     low = 0
            low = float(item["radio.outermost_level"])

            fitsfile = f"fits/{args.survey}/" + id + ".fits"

            # create the PNG:
            if os.path.exists(fitsfile):
                imgname = plot_image(fitsfile, ra, dec, majaxis, low)
                if imgname is not None:
                    n += 1

    return n


def make_csvfile(minsize=None, maxsize=None):
    filename = "static_rgz_flat_2019-05-06_full.csv"
    data = pd.read_csv(filename)

    n = 0
    df = None
    for i in tqdm(range(len(data))):
        item = data.iloc[i, :]
        majaxis = float(item["radio.max_angular_extent"])  # arcsec

        if minsize < majaxis < maxsize:
            id = str(item["rgz_name"])
            ra = float(item["radio.ra"])
            dec = float(item["radio.dec"])

            fitsfile = f"fits/{args.survey}/" + id + ".fits"
            imgfile = f"./img/{args.survey}/" + ".".join(fitsfile.split(".")[0:2]).split("/")[1] + ".png"

            isfits = os.path.exists(fitsfile)
            ispng = os.path.exists(imgfile)

            if isfits and ispng:
                match = match_mb(ra, dec, majaxis)
                df = update_csv(id, imgfile.split("/")[2], majaxis, match, df=df)
            else:
                print("Something is missing: {},{}".format(isfits, ispng))

    # Does dropna() need to be added here?
    df = df[~df.duplicated(["Source ID"])]  # remove duplicates from catalogue

    print(f"Data points: {len(df.index)} NaNs: {df.isnull().sum().sum()}")

    print("\n Writing to .csv file...")
    df.to_csv(f"RGZDR1Images.csv")

    return


def check_files(n, dir):
    files = os.listdir(dir)
    print("Number of files: {}".format(len(files)))
    print("Expected # of files: {}".format(n))

    return


if __name__ == "__main__":
    if args.dld:
        n = get_all_fits(minsize=15, maxsize=270)
        check_files(n, dir="fits")

    if args.img:
        n = get_all_imgs(minsize=15, maxsize=270)
        check_files(n, dir="img")

    if args.csv:
        make_csvfile(minsize=15, maxsize=270)
