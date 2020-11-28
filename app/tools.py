from pathlib import Path
import numpy as np

MAX_DIST = 30
BOX_HEIGHT_MARGIN = 30
BOX_WIDTH_MARGIN = 23
MAX_LETTER_INCIDENTS = 1000
NUM_OF_LETTERS = 20
CENTER_POINT = (600, 400)
MIN_MOVES = (-100, -100)
MAX_MOVES = (350, 800)


IMAGE_PATH = r"..\..\test_pages\test_gez good - Copy.jpg"
LETTERS_PATH = Path(r"..\..\test_pages\test_gez good")


def are_points_close(letter_location, location, dist=MAX_DIST):
    return np.linalg.norm(np.array(location) - np.array(letter_location)) < dist
