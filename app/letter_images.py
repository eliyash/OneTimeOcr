import tkinter as tk
from typing import Callable
import numpy as np

from PIL import ImageTk, Image

from app.data_model import ViewModel
from app.tools import SpecialGroupsEnum


class LettersImagesFrame:
    def __init__(
            self,
            view_model: ViewModel,
            run_gui_action: Callable,
            get_image_patch: Callable,
            frame
    ):
        self._letters_in_a_row = 10
        self._view_model = view_model
        self._run_gui_action = run_gui_action
        self._get_image_patch = get_image_patch
        self._frame = frame

        self._tk_image = ImageTk.PhotoImage(self._view_model.data_model.image)
        self._currentFrame = None
        self._create_new_frame()

    def _remove_images(self):
        self._currentFrame.destroy()
        self._create_new_frame()

    def _create_new_frame(self):
        self._currentFrame = tk.Frame(self._frame)
        self._currentFrame.grid(row=0, column=0)

    def _remove_letter(self, location):
        pass

    def _set_actions(self, label, location):
        pass

    def _del_image(self, label, location):
        label.destroy()
        self._remove_letter(location)

    def show_images(self, current_location_duplicates):
        self._remove_images()
        cv_image = np.array(self._view_model.data_model.image)
        for i, location in enumerate(current_location_duplicates):
            row = i // self._letters_in_a_row
            column = i % self._letters_in_a_row
            cv_letter_image = self._get_image_patch(cv_image, location)
            tk_letter_image = ImageTk.PhotoImage(Image.fromarray(cv_letter_image))
            label = tk.Label(self._currentFrame, image=tk_letter_image)
            label.image = tk_letter_image
            self._set_actions(label, location)
            label.grid(row=row, column=column)


class DuplicateLettersFrame(LettersImagesFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._view_model.current_location_duplicates.attach(self._run_gui_action(self.show_images))

    def _remove_letter(self, location):
        current_key = self._view_model.current_chosen_letter.data
        data = self._view_model.data_model.instances_locations_by_letters.data
        data[self._view_model.current_chosen_letter.data].remove(location)
        if current_key is not SpecialGroupsEnum.UNKNOWN:
            if SpecialGroupsEnum.UNKNOWN not in data:
                data[SpecialGroupsEnum.UNKNOWN] = set()
            data[SpecialGroupsEnum.UNKNOWN].add(location)
        self._view_model.data_model.instances_locations_by_letters.data = data

    def _set_actions(self, label, location):
        label.bind("<Button-3>", lambda e: self._del_image(label, location))


class MainLettersScreen(LettersImagesFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._view_model.data_model.instances_locations_by_letters.attach(self._run_gui_action(self.show_images))
        self._letters_in_a_row = 40

    def show_images(self, instances_locations_by_letters):
        super().show_images(list(instances_locations_by_letters.keys()))

    def _remove_letter(self, location):
        data = self._view_model.data_model.instances_locations_by_letters.data
        data.pop(location)
        self._view_model.data_model.instances_locations_by_letters.data = data

    def _set_main_letter(self, _, location):
        self._view_model.current_chosen_letter.data = location

    # def _f(self, e):
    #     release_point = e.widget.winfo_x() + e.x, e.widget.winfo_y() + e.y
    #     release_point
    #     for location, label in self._map.items():
    #         if are_points_close(location, release_point):
    #             label.destroy()

    def _set_actions(self, label, location):
        label.bind("<Button-1>", lambda e: self._set_main_letter(label, location))
        # label.bind("<ButtonRelease-1>", self._f)
        label.bind("<Button-3>", lambda e: self._del_image(label, location))
