from astropy.io import fits
from astroquery.skyview import SkyView
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.vizier import Vizier

import requests

import numpy as np
import pandas as pd
import pylab as pl
import os, sys
from tqdm import tqdm
from PIL import Image


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

    if not os.path.exists("fits"):
        os.mkdir("fits")

    fitsname = "fits/FIRST_" + id + ".fits"
    if overwrite == True or not os.path.exists(fitsname):

        sky = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
        url = SkyView.get_image_list(
            position=sky, survey=["VLA FIRST (1.4 GHz)"], cache=False
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

    data = np.squeeze(hdu[0].data)
    # pl.subplot(161)
    # pl.imshow(data)
    # pl.title("Orig")

    # crop centre:
    img = crop_centre(data, crop=150)
    # pl.subplot(162)
    # pl.imshow(img)
    # pl.title("Crop")

    # radial crop:
    img = apply_circular_mask(img, maj)
    # pl.subplot(163)
    # pl.imshow(img)
    # pl.title("Mask")

    # remove nans:
    img[np.where(np.isnan(img))] = 0.0
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
    pngname = "./img/" + ".".join(fitsfile.split(".")[0:2]).split("/")[1]
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
        df = df.append(df_tmp, ignore_index=True)
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
            result = Vizier.query_region(
                sky, radius=rad * u.arcsec, catalog=["J/MNRAS/466/4346/table1"]
            )
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


def get_all_imgs(minsize=None, maxsize=None):

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

            # Set low=0 to remove sigma clipping
            # low = float(item["radio.outermost_level"])
            low = 0

            fitsfile = "fits/FIRST_" + id + ".fits"

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

            fitsfile = "fits/FIRST_" + id + ".fits"
            imgfile = (
                "./img/" + ".".join(fitsfile.split(".")[0:2]).split("/")[1] + ".png"
            )

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
    df.to_csv("RGZDR1Images.csv")

    return


def check_files(n, dir):

    files = os.listdir(dir)
    print("Number of files: {}".format(len(files)))
    print("Expected # of files: {}".format(n))

    return


if __name__ == "__main__":

    dld_fits = True
    get_imgs = True
    make_cat = True

    if dld_fits:
        n = get_all_fits(minsize=15, maxsize=270)
        check_files(n, dir="fits")

    if get_imgs:
        n = get_all_imgs(minsize=15, maxsize=270)
        check_files(n, dir="img")

    if make_cat:
        make_csvfile(minsize=15, maxsize=270)
