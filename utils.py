import yaml

from pathlib import Path
from PIL import Image


def create_path(*args):
    for path in args:
        if not Path.exists(path):
            Path.mkdir(path, parents=True)


def array_to_png(img):
    """
    This function converts a numpy array to a png image and saves it to the given path.

    Args:
        img: Numpy array to convert to png
    """
    img = Image.fromarray(img)
    img = img.convert("L")

    return img


def load_config():
    """Helper function to load yaml config file, convert to python dictionary and return."""

    # load global config
    global_path = "config.yml"
    with open(global_path, "r") as ymlconfig:
        config = yaml.load(ymlconfig, Loader=yaml.FullLoader)

    survey_hashmap = {"vla_first": "VLA FIRST (1.4 GHz)", "nvss": "NVSS"}

    # Transcribe survey name for SkyView
    config["survey"] = survey_hashmap[config["survey"]]

    return config
