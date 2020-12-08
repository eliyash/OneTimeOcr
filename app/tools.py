from pathlib import Path
from typing import Dict
import cv2
import numpy as np

DATA_ROOT = Path(r"..\..\data")
IMAGES_PATH = DATA_ROOT / 'images'
LETTERS_PATH = DATA_ROOT / 'annotations'

MAX_DIST = 30
BOX_HEIGHT_MARGIN = 30
BOX_WIDTH_MARGIN = 23
MAX_LETTER_INCIDENTS = 1000
NUM_OF_LETTERS = 20
CENTER_POINT = (600, 400)
MIN_MOVES = (-100, -100)
MAX_MOVES = (350, 800)
ZERO_TRANSLATION = (0, 0)
PAGE_SIZE = (1500, 1500)


EMPTY_IMAGE = np.ones((BOX_HEIGHT_MARGIN*2, BOX_WIDTH_MARGIN*2)) * 255
UNKNOWN_IMAGE = cv2.line(
    cv2.circle(EMPTY_IMAGE.copy(), (BOX_HEIGHT_MARGIN, BOX_WIDTH_MARGIN), 15, 0, 2),
    (0, 0), (BOX_HEIGHT_MARGIN*2, BOX_WIDTH_MARGIN*2), 0, 2
)

UNKNOWN_KEY = 'unknown'


def are_points_close(letter_location, location, dist=MAX_DIST):
    return np.linalg.norm(np.array(location) - np.array(letter_location)) < dist


def get_values_to_add_and_remove(old: Dict, new: Dict):
    old_keys = set(old.keys())
    new_keys = set(new.keys())
    return old_keys-new_keys, new_keys-old_keys


_device = None


def get_device():
    global _device
    if not _device:
        import torch
        _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return _device
