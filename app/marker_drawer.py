from typing import Set

from app.data_model import ViewModel
from app.special_values import ZERO_X_Y
from app.tools import box_lines_from_center, points_sub


class MarkerDrawer:
    def __init__(self, view_model: ViewModel, canvas, color, box_shape, translator=None):
        self._local_instances_locations = set()
        self._view_model = view_model
        self._canvas = canvas
        self._color = color
        self._box_shape = box_shape
        self._translator = translator
        self._box_int_by_location = {}

    def __del__(self):
        self._update_letters(set())

    def _get_tag_from_location(self, location):
        return (location + (self._color,),)

    def _get_lines_of_point_in_canvas(self, location):
        translated_location = self._translator(location, inverse=True)
        return box_lines_from_center(translated_location, self._box_shape)

    def _is_in_screen(self, location):
        max_x, max_y = self._view_model.current_page_view_shape
        x_start, y_start, x_stop, y_stop = self._get_lines_of_point_in_canvas(location)
        return x_start >= 0 and y_start >= 0 and x_stop < max_x and y_stop < max_y

    def _add_box(self, location):
        box_int = self._canvas.create_rectangle(
            *ZERO_X_Y, *self._box_shape,
            outline='', fill='', stipple='gray50',
            tags=self._get_tag_from_location(location)
        )
        self._box_int_by_location[location] = box_int

    def _remove_box(self, location):
        box_int = self._box_int_by_location.pop(location)
        self._canvas.delete(box_int)

    def _update_box(self, location):
        box_int = self._box_int_by_location[location]
        if self._is_in_screen(location):
            new_location_start = self._get_lines_of_point_in_canvas(location)[:2]
            color = self._color
        else:
            new_location_start = ZERO_X_Y
            color = ''
        location_start_diff = points_sub(new_location_start, self._canvas.coords(box_int))
        self._canvas.move(box_int, *location_start_diff)
        self._canvas.itemconfig(box_int, fill=color, outline=color)

    def _update_letters(self, updated_letters: Set):
        updated_letters_locations = updated_letters.copy()
        letters_to_remove = self._local_instances_locations - updated_letters_locations
        letters_to_add = updated_letters_locations - self._local_instances_locations
        list(map(self._remove_box, letters_to_remove))
        list(map(self._add_box, letters_to_add))
        list(map(self._update_box, updated_letters_locations))
        self._local_instances_locations = updated_letters_locations


class SimpleMarkerDrawer(MarkerDrawer):
    def update_letters(self, updated_letters: Set):
        self._update_letters(updated_letters)
