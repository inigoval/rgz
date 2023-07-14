import os
import pickle
import numpy as np
import hashlib
import pandas as pd
import argparse

from pathlib import Path
from PIL import Image

from utils import load_config


parser = argparse.ArgumentParser(description="Parameters for batching")
parser.add_argument("-n", "--nbatch", type=int, default=54, help="# of batches")
parser.add_argument("-b", "--batchsize", type=int, default=2000, help="Batch size")
parser.add_argument("-i", "--imfolder", default="img/", help="Image folder")
parser.add_argument(
    "-su", "--survey", type=str, default="FIRST", help="Choose which survey to get data from"
)
args = parser.parse_args()


# -------------------------------------------------------------


def randomise_by_index(inputlist, idx_list):
    """
    Function to randomize an array of data
    """

    if len(inputlist) != len(idx_list):
        print("These aren't the same length")

    outputlist = []
    for i in idx_list:
        outputlist.append(inputlist[i])

    return outputlist


# -------------------------------------------------------------


def make_meta(path):
    class_labels = [""]

    # create dictionary of batch:
    dict = {
        "label_names": class_labels,
    }

    # write pickled output:
    with open(path / "batches" / "batches.meta", "wb") as f:
        pickle.dump(dict, f)

    return


# -------------------------------------------------------------


def make_batch(path, df, batch, n_batches, pbatch):
    png_dir = path / "PNG"
    batch_dir = path / "batches"

    if not Path.exists(batch_dir):
        Path.mkdir(batch_dir)

    if batch == (n_batches - 1):
        # the last batch is the test batch:
        oname = Path("test_batch")
        batch_label = "testing batch 1 of 1"
    else:
        # everything else is a training batch:
        oname = Path("data_batch_" + str(batch + 1))
        batch_label = Path("training batch " + str(batch + 1) + " of " + str(n_batches - 1))

    src_ids = df["Source ID"].to_numpy()
    files = df["Map File"].to_numpy()
    mbflag = df["MiraBest"].to_numpy()
    las = df["LAS"].to_numpy()
    ra = df["radio.ra"].to_numpy()
    dec = df["radio.dec"].to_numpy()

    # create empty arrays for the batches:
    ids = []
    data = []
    filenames = []
    mbflags = []
    sizes = []
    ra = []
    dec = []

    i0 = (pbatch * batch) - 1
    i = i0
    while True:
        i += 1
        if i >= (i0 + pbatch + 1) or i >= len(src_ids):
            break

        filename = files[i]
        flag = mbflag[i]
        size = las[i]

        if filename != "No file":
            filenames.append(filename)
            mbflags.append(flag)
            sizes.append(size)
            ra.append(ra[i])
            dec.append(dec[i])

            id = src_ids[i]
            ids.append(id)

            im = Image.open(png_dir + filename)
            im = np.array(im).flatten()
            filedata = np.array(list(im), np.uint8)
            data.append(filedata)

    print("Batched " + str(i - i0 - 1) + " files")

    if len(filenames) > 0:
        # randomise data in batch:
        idx_list = range(0, len(filenames))
        ids = randomise_by_index(ids, idx_list)
        data = randomise_by_index(data, idx_list)
        filenames = randomise_by_index(filenames, idx_list)
        mbflags = randomise_by_index(mbflags, idx_list)
        sizes = randomise_by_index(sizes, idx_list)
        ra = randomise_by_index(ra, idx_list)
        dec = randomise_by_index(dec, idx_list)

        # create dictionary of batch:
        dict = {
            "batch_label": batch_label,
            "data": data,
            "filenames": filenames,
            "src_ids": ids,
            "mb_flag": mbflags,
            "LAS": sizes,
            "ra": ra,
            "dec": dec,
        }

        # write pickled output:
        with open(batch_dir / oname, "wb") as f:
            pickle.dump(dict, f)

    return


# -------------------------------------------------------------


def make_batches(path, df, n_batches, batch_size):
    n_obj = len(df) - df[df["Map File"] == "No file"].shape[0]

    assert (n_obj) > (batch_size * (n_batches - 1))

    for batch in range(n_batches):
        make_batch(path, df, batch, n_batches, batch_size)

    return


# -------------------------------------------------------------


# -------------------------------------------------------------
# -------------------------------------------------------------

if __name__ == "__main__":
    config = load_config()

    n_data = int(args.nbatch * args.batchsize // 1000)
    batch_folder = Path(f"./rgz{n_data}k-{args.survey}-batches-py/")

    csvfile = "RGZDR1Images.csv"
    df = pd.read_csv(csvfile)

    dir = Path("rgz" / config["survey"])

    # make batched data:
    make_batches(dir, df, config["n_batches"], config["batch_size"])

    # make meta data:
    make_meta(dir)

    # get checksums:
    batchdir = batch_folder
    for file in Path.iterdir(batchdir):
        checksum = hashlib.md5(open(file, "rb").read()).hexdigest()
        print(file, checksum)

    # make tarfile:
    tarfile = f"rgz{n_data}k-batches-python.tar.gz"
    print("-----------------")
    print("Creating tarfile:")
    os.system("tar -cvzf " + tarfile + " " + str(batchdir) + "*batch* \n")
    print("-----------------")
    checksum = hashlib.md5(open(tarfile, "rb").read()).hexdigest()
    print("tgz_md5", checksum)
