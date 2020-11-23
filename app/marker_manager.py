from typing import Set, Callable

from app.observers import Subject
from app.tools import BOX_WIDTH_MARGIN, BOX_HEIGHT_MARGIN


class MarkerDrawer:
    def __init__(self, canvas, color, box_width_margin=BOX_WIDTH_MARGIN, box_height_margin=BOX_HEIGHT_MARGIN):
        self._local_instances_locations = set()
        self._canvas = canvas
        self._color = color
        self._box_width_margin = box_width_margin
        self._box_height_margin = box_height_margin

    def __del__(self):
        self._update_letters(set())

    def _add_a_box(self, location):
        x_center, y_center = location
        self._canvas.create_rectangle(
            x_center - self._box_width_margin,
            y_center - self._box_height_margin,
            x_center + self._box_width_margin,
            y_center + self._box_height_margin,
            tags=(location + (self._color,),),
            outline=self._color
        )

    def _remove_a_box(self, location):
        self._canvas.delete(location + (self._color,))

    def _update_letters(self, updated_letters: Set):
        letters_to_remove = self._local_instances_locations - updated_letters
        letters_to_add = updated_letters - self._local_instances_locations
        [self._remove_a_box(letter_to_remove) for letter_to_remove in letters_to_remove]
        [self._add_a_box(letter_to_add) for letter_to_add in letters_to_add]
        self._local_instances_locations = updated_letters.copy()


class MarkerManager(MarkerDrawer):
    def __init__(self, instances_locations_observer: Subject, run_gui_action: Callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._instances_locations = instances_locations_observer
        self._instances_locations.attach(run_gui_action(self._update_letters))

    def __del__(self):
        self._instances_locations.detach(self._update_letters)
        super().__del__()


class SimpleMarkerDrawer(MarkerDrawer):
    def update_letters(self, updated_letters: Set):
        self._update_letters(updated_letters)
