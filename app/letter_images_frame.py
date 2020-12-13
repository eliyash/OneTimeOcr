import tkinter as tk
from typing import Callable
from PIL import ImageTk, Image
from app.data_model import ViewModel
from app.special_values import UNKNOWN_KEY


class LettersImagesFrame:
    def __init__(self, view_model: ViewModel, run_gui_action: Callable, get_image_patch: Callable, frame):
        self._letters_in_a_row = 30
        self._view_model = view_model
        self._run_gui_action = run_gui_action
        self._get_image_patch = get_image_patch
        self._frame = frame

        self._map_keys_by_widgets = {}
        self._map_widgets_by_keys = {}
        self._currentFrame = tk.Frame(self._frame)
        self._currentFrame.grid(row=0, column=0)

    @property
    def _get_current_location_duplicates(self):
        return set(self._map_keys_by_widgets.values())

    def _set_actions(self, label, location):
        pass

    def update_images(self, new_location_duplicates: set):
        current_location_duplicates = self._get_current_location_duplicates
        new_location_duplicates = set(new_location_duplicates)
        for key_to_add in new_location_duplicates - current_location_duplicates:
            cv_letter_image = self._get_image_patch(key_to_add)
            tk_letter_image = ImageTk.PhotoImage(Image.fromarray(cv_letter_image))
            label = tk.Label(self._currentFrame, image=tk_letter_image)
            label.image = tk_letter_image
            self._set_actions(label, key_to_add)
            self._map_keys_by_widgets[label] = key_to_add
            self._map_widgets_by_keys[key_to_add] = label

        for key_to_remove in current_location_duplicates - new_location_duplicates:
            label = self._map_widgets_by_keys.pop(key_to_remove)
            self._map_keys_by_widgets.pop(label)
            label.destroy()

        for i, label in enumerate(self._map_widgets_by_keys.values()):
            row = i // self._letters_in_a_row
            column = i % self._letters_in_a_row
            label.grid(row=row, column=column)


class DuplicateLettersFrame(LettersImagesFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._view_model.current_location_duplicates.attach(self._run_gui_action(self.update_images))

    def _remove_letter(self, location):
        data = self._view_model.data_model.instances_locations_by_letters.data
        data[self._view_model.current_chosen_letter.data].remove(location)
        self._view_model.data_model.instances_locations_by_letters.data = data

    def _clear_letter_key(self, location):
        current_key = self._view_model.current_chosen_letter.data
        self._remove_letter(location)
        if current_key is not UNKNOWN_KEY:
            self._add_letter(UNKNOWN_KEY, location)

    def _try_move_letter(self, event, location):
        widget = event.widget.winfo_containing(event.x_root, event.y_root)
        if widget and widget in self._view_model.map_keys_by_widgets:
            hovered_key = self._view_model.map_keys_by_widgets[widget]
            self._remove_letter(location)
            self._add_letter(hovered_key, location)

    def _add_letter(self, key, location):
        data = self._view_model.data_model.instances_locations_by_letters.data
        data[key].add(location)
        self._view_model.data_model.instances_locations_by_letters.data = data

    def _set_actions(self, label, location):
        label.bind("<Button-3>", lambda e: self._clear_letter_key(location))
        label.bind("<ButtonRelease-1>", lambda e: self._try_move_letter(e, location))


class MainLettersScreen(LettersImagesFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._view_model.data_model.different_letters.attach(self._run_gui_action(self.show_images))
        self._view_model.current_chosen_letter.attach(self._run_gui_action(self.update_chosen_letter))
        self._letters_in_a_row = 40

    def update_chosen_letter(self, _):
        different_letters = self._get_current_location_duplicates
        super().update_images(set())
        self.show_images({k: None for k in different_letters})

    def show_images(self, different_letters):
        super().update_images(set(different_letters.keys()))
        self._view_model.map_keys_by_widgets = self._map_keys_by_widgets

    def _remove_letter(self, location):
        if location == UNKNOWN_KEY:
            self.reset_unknown_letters()
        else:
            self.remove_letter(location)

    def remove_letter(self, location):
        data = self._view_model.data_model.different_letters.data
        data.pop(location)
        self._view_model.data_model.different_letters.data = data

    def reset_unknown_letters(self):
        data = self._view_model.data_model.instances_locations_by_letters.data
        data[UNKNOWN_KEY] = set()
        self._view_model.data_model.instances_locations_by_letters.data = data

    def _set_main_letter(self, location):
        self._view_model.current_chosen_letter.data = location

    def _set_actions(self, label, location):
        label.bind("<Button-3>", lambda e: self._remove_letter(location))
        label.bind("<Button-1>", lambda e: self._set_main_letter(location))
