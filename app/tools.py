from pathlib import Path
from typing import Dict, Tuple, Set
import cv2
import numpy as np

DATA_ROOT = Path(r"../../data")
IMAGES_PATH = DATA_ROOT / 'images'
LETTERS_PATH = DATA_ROOT / 'annotations'

NETWORK_ROOT = Path(r"../networks")
IDENTIFIER_NETS_PATH = NETWORK_ROOT / 'identifier'
DETECTOR_NETS_PATH = NETWORK_ROOT / 'detector'

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
LAST_NET = 'last_net.pth'

EMPTY_IMAGE = np.ones((BOX_HEIGHT_MARGIN*2, BOX_WIDTH_MARGIN*2)) * 255
UNKNOWN_IMAGE = cv2.line(
    cv2.circle(EMPTY_IMAGE.copy(), (BOX_HEIGHT_MARGIN, BOX_WIDTH_MARGIN), 15, 0, 2),
    (0, 0), (BOX_HEIGHT_MARGIN*2, BOX_WIDTH_MARGIN*2), 0, 2
)

UNKNOWN_KEY = 'unknown'


def location_to_str(location):
    if location != UNKNOWN_KEY:
        return '_'.join(list(location))
    else:
        return location


def str_to_location(location):
    if location != UNKNOWN_KEY:
        return tuple(map(int, location.split('_')))
    else:
        return location


def are_points_close(letter_location, location, dist=MAX_DIST):
    return np.linalg.norm(np.array(location) - np.array(letter_location)) < dist


def is_different_values_preset(old: Dict, new: Dict) -> bool:
    return len(set(old.keys()).symmetric_difference(set(new.keys()))) > 0


def get_values_to_add_and_remove(old: Dict, new: Dict) -> Tuple[Set, Set]:
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


def file_name_to_location(file_name):
    return str_to_location(file_name.name.split('.')[0])


def is_non_last_net(path: Path):
    return path.name.endswith('.pth') and path.name != LAST_NET


def get_last_epoch_net(network_path):
    model_path = max(
        [path for path in network_path.iterdir() if is_non_last_net(path)],
        key=lambda path: int(path.name.split('_')[-1].split('.')[0])
    )
    return model_path
