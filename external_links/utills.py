import pathlib
from io import BytesIO
import requests
from zipfile import ZipFile
import pandas as pd

PROJECT_DIR = pathlib.Path("..")
DATA_DIR = pathlib.Path("../data")


def get_el_dataset():
    url_str = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip'
    file_content = requests.get(url_str).content
    file_zipped = ZipFile(BytesIO(file_content))
    csv_unzipped = file_zipped.open('LD2011_2014.txt')
    return pd.read_csv(csv_unzipped)


def download_dataset():
    url_str = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip'
    zip_out = DATA_DIR/"LD2011_2014.txt.zip"
    DATA_DIR.mkdir(exist_ok=True)
    file_content = requests.get(url_str).content
    with open(zip_out, 'wb') as out_file:
        out_file.write(file_content)
    with ZipFile(zip_out, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
