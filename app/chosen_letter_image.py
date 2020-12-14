import tkinter as tk
from typing import Callable

from PIL import ImageTk, Image

from app.data_model import ViewModel


class ChosenLetterImageHandler:
    def __init__(self, view_model: ViewModel, run_gui_action: Callable, top_bar, empty_image):
        self._view_model = view_model
        self._empty_image = empty_image

        self._chosen_letter_image = tk.Label(top_bar)
        self._chosen_letter_image.pack(side=tk.LEFT)

        self._set_chosen_letter_image(None)
        self._view_model.current_chosen_letter.attach(run_gui_action(self._set_chosen_letter_image))

    def _set_chosen_letter_image(self, letter):
        if letter in self._view_model.data_model.different_letters.data:
            cv_letter_image = self._view_model.data_model.different_letters.data[letter]
        else:
            cv_letter_image = self._empty_image
        tk_letter_image = ImageTk.PhotoImage(Image.fromarray(cv_letter_image))
        self._chosen_letter_image.config(image=tk_letter_image)
        self._chosen_letter_image.image = tk_letter_image
