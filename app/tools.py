from enum import Enum
from pathlib import Path
import cv2
import numpy as np

DATA_ROOT = Path(r"..\..\data")
IMAGES_PATH = DATA_ROOT / 'images'
LETTERS_PATH = DATA_ROOT / 'letters'

MAX_DIST = 30
BOX_HEIGHT_MARGIN = 30
BOX_WIDTH_MARGIN = 23
MAX_LETTER_INCIDENTS = 1000
NUM_OF_LETTERS = 20
CENTER_POINT = (600, 400)
MIN_MOVES = (-100, -100)
MAX_MOVES = (350, 800)

EMPTY_IMAGE = np.ones((BOX_HEIGHT_MARGIN*2, BOX_WIDTH_MARGIN*2)) * 255
UNKNOWN_IMAGE = cv2.line(
    cv2.circle(EMPTY_IMAGE, (BOX_HEIGHT_MARGIN, BOX_WIDTH_MARGIN), 15, 0, 2),
    (0, 0), (BOX_HEIGHT_MARGIN*2, BOX_WIDTH_MARGIN*2), 0, 2
)


class SpecialGroupsEnum(Enum):
    UNKNOWN = 0


def are_points_close(letter_location, location, dist=MAX_DIST):
    return np.linalg.norm(np.array(location) - np.array(letter_location)) < dist
