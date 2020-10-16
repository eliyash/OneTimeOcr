import cv2
import numpy as np
from typing import Dict, Tuple
from app.gui import Gui
from app.tools import BOX_WIDTH_MARGIN, BOX_HEIGHT_MARGIN, are_points_close, IMAGE_PATH, LETTERS_PATH, \
    MAX_LETTER_INCIDENTS, NUM_OF_LETTERS


class App:
    def __init__(self):
        self._gui = Gui(IMAGE_PATH, LETTERS_PATH, self._on_save_letters, self._on_save_letters)

    def _on_save_letters(self, letters_centers: Tuple):
        image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE).astype('float16') / 256
        for key in letters_centers:
            (x_center, y_center) = key
            letter_image = image[
                y_center - BOX_HEIGHT_MARGIN:
                y_center + BOX_HEIGHT_MARGIN,
                x_center - BOX_WIDTH_MARGIN:
                x_center + BOX_WIDTH_MARGIN
            ]

            nw_locations = self._basic_matching_new(image, letter_image)
            nw_locations.sort(key=lambda v: v[1], reverse=True)
            to_add = []
            to_check = nw_locations
            while to_check:
                new_point = to_check.pop(0)
                to_add.append(new_point)
                to_check = [location for location in to_check if not are_points_close(location[0], new_point[0])]

            to_add = to_add[:NUM_OF_LETTERS]
            locations = [(x_center + BOX_WIDTH_MARGIN, y_center + BOX_HEIGHT_MARGIN)
                         for (y_center, x_center), val in to_add]
            self._gui.set_duplicate_letters(key, locations)

    @staticmethod
    def _basic_matching_new(image, letter_patch, number_letters=MAX_LETTER_INCIDENTS):
        res = cv2.matchTemplate(image, letter_patch, cv2.TM_CCORR_NORMED)
        matching_values = np.sort(res, axis=None, kind=None, order=None)

        min_value = matching_values[-number_letters]
        above_mat = res > min_value
        locations_by_values = [
            [(x_center, y_center), res[x_center, y_center]]
            for x_center, y_center in np.transpose(np.nonzero(above_mat))
        ]
        return locations_by_values

    def run(self):
        self._gui.run()


if __name__ == '__main__':
    App().run()
