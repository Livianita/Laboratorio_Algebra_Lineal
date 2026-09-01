import os
import urllib.request
import zipfile

import numpy as np
from PIL import Image

DATASET_DIR = os.path.join('lfwcrop_grey', 'faces')
VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.pgm'}


def ensure_dataset():
    if os.path.isdir(DATASET_DIR):
        return

    url = 'http://conradsanderson.id.au/lfwcrop/lfwcrop_grey.zip'
    zip_path = 'lfwcrop_grey.zip'

    print('Descargando dataset...')
    urllib.request.urlretrieve(url, zip_path)

    print('Extrayendo dataset...')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall('.')


def load_images():
    ensure_dataset()

    filenames = []
    images = []

    for filename in os.listdir(DATASET_DIR):
        file_path = os.path.join(DATASET_DIR, filename)

        if not os.path.isfile(file_path):
            continue

        if os.path.splitext(filename)[1].lower() not in VALID_EXTENSIONS:
            continue

        filenames.append(filename)
        images.append(np.array(Image.open(file_path)))

    return filenames, np.array(images)


def build_matrix(images):
    n = 64 * 64
    return images.reshape(len(images), n)
