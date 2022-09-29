import os, sys
import pickle
import numpy as np
from PIL import Image
import hashlib
import pandas as pd

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


def make_meta(bindir="./rgz20k-batches-py/"):

    oname = "batches.meta"

    class_labels = [""]

    # create dictionary of batch:
    dict = {
        "label_names": class_labels,
    }

    # write pickled output:
    with open(bindir + oname, "wb") as f:
        pickle.dump(dict, f)

    return


# -------------------------------------------------------------


def make_batch(df, batch, nbatch, pbatch, imdir="./img/", bindir="./rgz20k-batches-py/"):

    if not os.path.exists(bindir):
        os.mkdir(bindir)

    if batch == (nbatch - 1):
        # the last batch is the test batch:
        oname = "test_batch"
        batch_label = "testing batch 1 of 1"
    else:
        # everything else is a training batch:
        oname = "data_batch_" + str(batch + 1)
        batch_label = "training batch " + str(batch + 1) + " of " + str(nbatch - 1)

    src_ids = df["Source ID"].to_numpy()
    files = df["Map File"].to_numpy()
    mbflag = df["MiraBest"].to_numpy()
    las = df["LAS"].to_numpy()

    # create empty arrays for the batches:
    ids = []
    data = []
    filenames = []
    mbflags = []
    sizes = []

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

            id = src_ids[i]
            ids.append(id)

            im = Image.open(imdir + filename)
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

        # create dictionary of batch:
        dict = {
            "batch_label": batch_label,
            "data": data,
            "filenames": filenames,
            "src_ids": ids,
            "mb_flag": mbflags,
            "LAS": sizes,
        }

        # write pickled output:
        with open(bindir + oname, "wb") as f:
            pickle.dump(dict, f)

    return


# -------------------------------------------------------------


def make_batches(df, imdir="./img/"):

    nbatch = 11
    pbatch = 2000

    n_obj = len(df) - df[df["Map File"] == "No file"].shape[0]

    assert (n_obj) > (pbatch * (nbatch - 1))

    for batch in range(nbatch):
        make_batch(df, batch, nbatch, pbatch, imdir)

    return


# -------------------------------------------------------------


# -------------------------------------------------------------
# -------------------------------------------------------------

if __name__ == "__main__":

    csvfile = "RGZDR1Images.csv"
    df = pd.read_csv(csvfile)

    # make batched data:
    make_batches(df, imdir="./img/")

    # make meta data:
    make_meta()

    # get checksums:
    batchdir = "./rgz20k-batches-py/"
    for file in os.listdir(batchdir):
        checksum = hashlib.md5(open(batchdir + file, "rb").read()).hexdigest()
        print(file, checksum)

    # make tarfile:
    tarfile = "rgz20k-batches-python.tar.gz"
    print("-----------------")
    print("Creating tarfile:")
    os.system("tar -cvzf " + tarfile + " " + batchdir + "*batch* \n")
    print("-----------------")
    checksum = hashlib.md5(open(tarfile, "rb").read()).hexdigest()
    print("tgz_md5", checksum)
