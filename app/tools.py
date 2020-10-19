from pathlib import Path
import numpy as np

MAX_DIST = 30
BOX_HEIGHT_MARGIN = 30
BOX_WIDTH_MARGIN = 23
MAX_LETTER_INCIDENTS = 1000
NUM_OF_LETTERS = 20


IMAGE_PATH = r"C:\Workspace\MyOCR\EAST\test_pages\Screenshot 2020-10-11 232042.jpg"
LETTERS_PATH = Path(r"C:\Workspace\MyOCR\EAST\test_pages\letters_app")
# IMAGE_PATH = r"C:\Workspace\MyOCR\identifieng letters\data\books\handwriting1.jpg"
# LETTERS_PATH = Path(r"C:\Workspace\MyOCR\EAST\test_pages\hand")


def are_points_close(letter_location, location):
    return np.linalg.norm(np.array(location) - np.array(letter_location)) < MAX_DIST
