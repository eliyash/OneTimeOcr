import tkinter as tk
from typing import Callable
import numpy as np

from PIL import ImageTk, Image

from app.data_model import DataModel


class LettersImagesFrame:
    def __init__(
            self,
            data_model: DataModel,
            get_image_patch: Callable,
            frame
    ):
        self._letters_in_a_row = 10
        self._data_model = data_model
        self._get_image_patch = get_image_patch
        self._frame = frame

        self._tk_image = ImageTk.PhotoImage(self._data_model.image)
        self._currentFrame = None
        self._create_new_frame()
        self._data_model.current_location_duplicates.attach(self.show_images)

    def _remove_images(self):
        self._currentFrame.destroy()
        self._create_new_frame()

    def _create_new_frame(self):
        self._currentFrame = tk.Frame(self._frame)
        self._currentFrame.grid(row=0, column=0)

    @staticmethod
    def _get_del_image(label):
        return lambda e: label.destroy()

    def show_images(self, current_location_duplicates):
        self._remove_images()
        cv_image = np.array(self._data_model.image)
        for i, location in enumerate(current_location_duplicates):
            row = i // self._letters_in_a_row
            column = i % self._letters_in_a_row
            cv_letter_image = self._get_image_patch(cv_image, location)
            tk_letter_image = ImageTk.PhotoImage(Image.fromarray(cv_letter_image))
            label = tk.Label(self._currentFrame, image=tk_letter_image)
            label.bind("<Button-1>", self._get_del_image(label))
            label.image = tk_letter_image
            label.grid(row=row, column=column)
