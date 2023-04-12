from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity


class RGZ20k(data.Dataset):
    """`RGZ 20k <>`_Dataset

    Args:
        root (string): Root directory of dataset where directory
            ``htru1-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "rgz20k-batches-py"
    url = "http://www.jb.man.ac.uk/research/ascaife/rgz20k-batches-python.tar.gz"
    filename = "rgz20k-batches-python.tar.gz"
    tgz_md5 = "3a85fb4167fb08619e36e77bbba40896"
    train_list = [
        ["data_batch_1", "9ffdcb485fc0c96e1afa1cdc342f00e7"],
        ["data_batch_2", "8961fa4a2fb5a8482ec5606e3d501fb4"],
        ["data_batch_3", "8cbf4fa7b34282b8f1522a350df4a882"],
        ["data_batch_4", "a58c94b5905d0ad2d97ba3a8895538c9"],
        ["data_batch_5", "13c9132ee374b7b63dac22c17a412d86"],
        ["data_batch_6", "232ff5854df09d5a68471861b1ee5576"],
        ["data_batch_7", "bea739fe0f7bd6ffb77b8e7def7f2edf"],
        ["data_batch_8", "48b23b3f0f37478b61ccca462fe53917"],
        ["data_batch_9", "2ac1208dec744cc136d0cd7842c180a2"],
        ["data_batch_10", "4ad68d1d8179da93ca1f8dfa7fe8e11d"],
    ]

    test_list = [
        ["test_batch", "35a588227816ad08d37112f23b6e2ea4"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "8f3138fbc912134239a779a1f3f6eaf8",
    }

    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []  # object image data
        self.names = []  # object file names
        self.rgzid = []  # object RGZ ID
        self.mbflg = []  # object MiraBest flag
        self.sizes = []  # object largest angular sizes

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)

            with open(file_path, "rb") as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding="latin1")

                # print(entry.keys())

                self.data.append(entry["data"])
                self.names.append(entry["filenames"])
                self.rgzid.append(entry["src_ids"])
                self.mbflg.append(entry["mb_flag"])
                self.sizes.append(entry["LAS"])

        self.rgzid = np.vstack(self.rgzid).reshape(-1)
        self.sizes = np.vstack(self.sizes).reshape(-1)
        self.mbflg = np.vstack(self.mbflg).reshape(-1)
        self.names = np.vstack(self.names).reshape(-1)

        self.data = np.vstack(self.data).reshape(-1, 1, 150, 150)
        self.data = self.data.transpose((0, 2, 3, 1))

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError(
                "Dataset metadata file not found or corrupted."
                + " You can use download=True to download it"
            )
        with open(path, "rb") as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding="latin1")

            self.classes = data[self.meta["key"]]

        # self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            image (array): Image
        """

        img = self.data[index]
        las = self.sizes[index]
        mbf = self.mbflg[index]
        rgz = self.rgzid[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = np.reshape(img, (150, 150))
        img = Image.fromarray(img, mode="L")

        if self.transform is not None:
            img = self.transform(img)

        return img, rgz, las, mbf

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        tmp = "train" if self.train is True else "test"
        fmt_str += "    Split: {}\n".format(tmp)
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str


class RGZ108kk(data.Dataset):
    """`RGZ 20k <>`_Dataset

    Args:
        root (string): Root directory of dataset where directory
            ``htru1-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "rgz108k-batches-py"

    # Need to upload this, for now downloadis commented out
    # url = "http://www.jb.man.ac.uk/research/ascaife/rgz20k-batches-python.tar.gz"
    filename = "rgz108k-batches-python.tar.gz"
    tgz_md5 = "2447ac72f83f2ed146d5078a18595ff1"
    train_list = [
        ["data_batch_1", "0ddae9401bf06ec8160695667c20bc8e"],
        ["data_batch_2", "0b5c9e66f9e843610c46c86711122072"],
        ["data_batch_3", "82b3114440c345d14c268c610a6ef51a"],
        ["data_batch_4", "b5609c2d83005bc33d493af0cbe3f8ce"],
        ["data_batch_5", "921505741d513a8c9bd76aee5ec06ee8"],
        ["data_batch_6", "01c45d73fe996cfdf58ca25232731295"],
        ["data_batch_7", "a0a685bd626b3497fd1cf2e3543c0e39"],
        ["data_batch_8", "d098bd61ae99e4db8facab0ea1615482"],
        ["data_batch_9", "9ec99f4033f57dc699641a9ae412c0a7"],
        ["data_batch_10", "f9b4aeb245bfbc7e5b906ff525348cfb"],
        ["data_batch_11", "fa0268c669497bae162c7b074a94fb72"],
        ["data_batch_12", "3b9a4174ca26534e1befae5626c87b55"],
        ["data_batch_13", "8d5d4ab46b4266525dab490dc5c2ee7c"],
        ["data_batch_14", "66fd9d3a16cab324ef5eefa7f144a569"],
        ["data_batch_15", "03c4d031d1c5124eab7983dd632df1c8"],
        ["data_batch_16", "b9ecb1915257de9af49c1d96afc98f73"],
        ["data_batch_17", "03783f12d202b3d453ef59284b66427d"],
        ["data_batch_18", "394c78239a9cdce0a7658c982bf18dd2"],
        ["data_batch_19", "e5d6aa35fd1de066273282ca055c1fa3"],
        ["data_batch_20", "7c1e65d110cc069383e1b359eca24c2e"],
        ["data_batch_21", "d9473de896765e2d90ca45aae39a7d7b"],
        ["data_batch_22", "7e4bcbf2d5973e11771071fc9e8e5a7a"],
        ["data_batch_23", "5cfe7e992c81df8e7a0d9d30439f1337"],
        ["data_batch_24", "9ecdcff509498fbf31cac5fa3878ba64"],
        ["data_batch_25", "d1a978fc7f2b45947269e2344a354cdd"],
        ["data_batch_26", "29e7fe6456ded86350b2e165ba67a526"],
        ["data_batch_27", "930cc87218ec5614270f79b62d4b0ee0"],
        ["data_batch_28", "58ac0766de719ee77706a84b908017eb"],
        ["data_batch_29", "dfbea2e251497f0d9f1087af0d36ab31"],
        ["data_batch_30", "67ac5bafe8c635016437f056a04f6efe"],
        ["data_batch_31", "826434e598a12ce40d362fff05c85009"],
        ["data_batch_32", "4042794cd6029942a5f936d07055f43c"],
        ["data_batch_33", "58470786b95d4e534fbf62c3074de289"],
        ["data_batch_34", "f1e7334a072409e0ee4170383486212b"],
        ["data_batch_35", "80d9cef57668d87c35ec49f2e35027f2"],
        ["data_batch_36", "02bd891e99979f7d99f040788d314258"],
        ["data_batch_37", "f6ce3df967c40b557774f04776b70c3d"],
        ["data_batch_38", "aaee87de408447e86528c3af066b5358"],
        ["data_batch_39", "3c05fa04460c5b1bb7608813316081f0"],
        ["data_batch_40", "fda92f52155948022bfc8cf42f9079df"],
        ["data_batch_41", "91a926123ac8273340fff235ef0603d7"],
        ["data_batch_42", "827174c3cddfa36281b90a32d3cc073a"],
        ["data_batch_43", "74a276c8920d0a88e379e0820ba0eeb8"],
        ["data_batch_44", "e940fdfba4534115a5e0b0f3d98fd417"],
        ["data_batch_45", "8213c09c3942847c7161b6fc82f2a318"],
        ["data_batch_46", "a86410db097c35137d238f7e9a6de772"],
        ["data_batch_47", "58f96548dbadfa5382e7ac99ab19faa9"],
        ["data_batch_48", "0e52a942c8335387d80051fd90a69ad1"],
        ["data_batch_49", "6d4241a0dc4df07a0275e3cff66370f9"],
        ["data_batch_50", "290159924449740fad344973c10a3561"],
        ["data_batch_51", "e3ec98a0e4961e7724e8d12bf09a20b0"],
        ["data_batch_52", "3da1e10c4e820b7f10c64a8bc4d01f86"],
        ["data_batch_53", "269a5f200e9234fc0f66986e2652c865"],
    ]

    test_list = [
        ["test_batch", "51a72d607d501987ede7810b1441b7ec"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "d5d3d04e1d462b02b69285af3391ba25",
    }

    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        # if download:
        #     self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []  # object image data
        self.names = []  # object file names
        self.rgzid = []  # object RGZ ID
        self.mbflg = []  # object MiraBest flag
        self.sizes = []  # object largest angular sizes

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)

            with open(file_path, "rb") as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding="latin1")

                # print(entry.keys())

                self.data.append(entry["data"])
                self.names.append(entry["filenames"])
                self.rgzid.append(entry["src_ids"])
                self.mbflg.append(entry["mb_flag"])
                self.sizes.append(entry["LAS"])

        self.rgzid = np.vstack(self.rgzid).reshape(-1)
        self.sizes = np.vstack(self.sizes).reshape(-1)
        self.mbflg = np.vstack(self.mbflg).reshape(-1)
        self.names = np.vstack(self.names).reshape(-1)

        self.data = np.vstack(self.data).reshape(-1, 1, 150, 150)
        self.data = self.data.transpose((0, 2, 3, 1))

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError(
                "Dataset metadata file not found or corrupted."
                + " You can use download=True to download it"
            )
        with open(path, "rb") as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding="latin1")

            self.classes = data[self.meta["key"]]

        # self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            image (array): Image
        """

        img = self.data[index]
        las = self.sizes[index]
        mbf = self.mbflg[index]
        rgz = self.rgzid[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = np.reshape(img, (150, 150))
        img = Image.fromarray(img, mode="L")

        if self.transform is not None:
            img = self.transform(img)

        return img, rgz, las, mbf

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        tmp = "train" if self.train is True else "test"
        fmt_str += "    Split: {}\n".format(tmp)
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str
