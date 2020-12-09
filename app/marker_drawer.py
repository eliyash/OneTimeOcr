from typing import Set
from app.special_values import BOX_WIDTH_MARGIN, BOX_HEIGHT_MARGIN


class MarkerDrawer:
    def __init__(
            self, canvas, color, box_width_margin=BOX_WIDTH_MARGIN, box_height_margin=BOX_HEIGHT_MARGIN, translator=None
    ):
        self._local_instances_locations = set()
        self._canvas = canvas
        self._color = color
        self._box_width_margin = box_width_margin
        self._box_height_margin = box_height_margin
        self._translator = translator

    def __del__(self):
        self._update_letters(set())

    def _add_a_box(self, location):
        x_center, y_center = self._translator(location, inverse=True)
        self._canvas.create_rectangle(
            x_center - self._box_width_margin,
            y_center - self._box_height_margin,
            x_center + self._box_width_margin,
            y_center + self._box_height_margin,
            tags=(location + (self._color,),),
            outline=self._color,
            fill=self._color,
            stipple='gray50'
        )

    def _remove_a_box(self, location):
        self._canvas.delete(location + (self._color,))

    def _update_letters(self, updated_letters: Set):
        updated_letters_locations = {letter for letter in updated_letters if type(letter) is tuple}
        letters_to_remove = self._local_instances_locations - updated_letters_locations
        letters_to_add = updated_letters_locations - self._local_instances_locations
        [self._remove_a_box(letter_to_remove) for letter_to_remove in letters_to_remove]
        [self._add_a_box(letter_to_add) for letter_to_add in letters_to_add]
        self._local_instances_locations = updated_letters_locations


class SimpleMarkerDrawer(MarkerDrawer):
    def update_letters(self, updated_letters: Set):
        self._update_letters(updated_letters)
