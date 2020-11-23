import tkinter as tk
from tkinter.filedialog import askdirectory
from typing import Callable, Tuple

from PIL import ImageTk

from app.data_model import DataModel
from app.letter_images import LettersImagesFrame
from app.main_letters_handler import MainLettersHandler
from app.tools import are_points_close, NUM_OF_LETTERS, CENTER_POINT


class Gui:
    def __init__(
            self,
            data_model: DataModel,
            get_image_patch: Callable,
            on_look_for_letter_callback: Callable,
            network_detect_callback: Callable,
            save_letters_callback: Callable,
            get_images_by_locations_callback: Callable,
            translation: Tuple = (0, 0)
    ):
        self._data_model = data_model
        self._get_image_patch = get_image_patch
        self._on_look_for_letter_callback = on_look_for_letter_callback
        self._network_detect_callback = network_detect_callback
        self._save_letters_callback = save_letters_callback
        self._get_images_by_locations_callback = get_images_by_locations_callback
        self._translation = translation

        self._window = tk.Tk()
        self._tk_image = ImageTk.PhotoImage(self._data_model.image)
        self._top_bar = tk.Frame(self._window)
        self._top_bar.grid(row=0, column=0, sticky="nsew")

        self._save_button = tk.Button(self._top_bar, text="save lettres", command=self._on_save_all_letters)
        self._save_button.pack(side=tk.LEFT)

        self._look_for_dup_button = tk.Button(self._top_bar, text="look for letter", command=self._on_look_for_letter)
        self._look_for_dup_button.pack(side=tk.LEFT)

        self._call_net_button = tk.Button(self._top_bar, text="call net", command=self._network_detect_callback)
        self._call_net_button.pack(side=tk.LEFT)

        self._text_frame = tk.Frame(self._window)
        self._text_frame.grid(row=1, column=0, sticky="nsew")

        self._letters_frame = tk.Frame(self._window)
        self._letters_frame.grid(row=1, column=0, sticky="nsew")

        width, height = self._data_model.image.size
        self._canvas = tk.Canvas(self._text_frame, width=width, height=height)
        self._canvas.pack(side=tk.LEFT)
        self._canvas_image = self._canvas.create_image(0, 0, image=self._tk_image, anchor=tk.NW)

        self._canvas.bind("<Button-1>", self._on_mouse_press_left)
        self._canvas.bind("<Button-2>", self._on_mouse_press_wheel)
        self._canvas.bind("<Button-3>", self._on_mouse_press_right)

        x, y = CENTER_POINT
        self._canvas_center = self._canvas.create_rectangle(
            x-2, y-2, x+2, y+2, tags=('center',), outline='purple', width=5
        )

        self._duplicates = tk.Scale(self._top_bar, from_=1, to=200, orient=tk.HORIZONTAL)
        self._duplicates.set(NUM_OF_LETTERS)
        self._duplicates.pack(side=tk.LEFT)

        self._main_letters_handler = MainLettersHandler(
            self._data_model, self._run_gui_action, self._top_bar, self._canvas, self._get_image_patch, self._translator
        )

        self._main_letters_frame = LettersImagesFrame(
            self._data_model, self._run_gui_action, self._get_image_patch, self._letters_frame
        )

        self._switch_mode = tk.Button(self._top_bar, text="switch mode", command=self._on_switch_apps)
        self._switch_mode.pack(side=tk.LEFT)

        self._is_normal_mode = True
        self._text_frame.tkraise()

    def _translator(self, location, inverse=False):
        return tuple(axis + offset*(-1 if inverse else 1) for axis, offset in zip(location, self._translation))

    def _run_gui_action(self, func, delay=0):
        return lambda *args, **kwargs: self._window.after(delay, func(*args, **kwargs))

    def _on_save_all_letters(self):
        folder = askdirectory()
        self._save_letters_callback(folder, self._data_model.instances_locations_by_letters.data)

    def _on_look_for_letter(self):
        self._on_look_for_letter_callback(self._data_model.current_main_letter.data, self._duplicates.get())

    def _on_mouse_press_left(self, event):
        location = self._translator((event.x, event.y))
        self._main_letters_handler.add_main_letter(location)

    def _on_mouse_press_wheel(self, event):
        x, y = CENTER_POINT
        location = (event.x-x, event.y-y)
        self._canvas.move(self._canvas_image, *[-axis for axis in location])
        self._translation = self._translator(location)
        current_main_letter = self._data_model.current_main_letter.data
        main_letters = self._data_model.main_letters.data.copy()
        instances_locations_by_letters = {k: v for k, v in self._data_model.instances_locations_by_letters.data.items()}
        self._data_model.main_letters.data = set()
        self._data_model.main_letters.data = main_letters
        self._data_model.instances_locations_by_letters.data = instances_locations_by_letters
        self._data_model.current_main_letter.data = current_main_letter

    def _on_mouse_press_right(self, event):
        location = self._translator((event.x, event.y))
        letters_locations = self._data_model.main_letters.data
        for letter_location in letters_locations.copy():
            if are_points_close(letter_location, location):
                letters_locations.remove(letter_location)
        self._data_model.main_letters.data = letters_locations

    def _on_switch_apps(self):
        if self._is_normal_mode:
            self._letters_frame.tkraise()
        else:
            self._text_frame.tkraise()

        self._is_normal_mode = not self._is_normal_mode

    def run(self):
        self._window.mainloop()
