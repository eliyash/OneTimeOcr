import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog
from typing import Callable, Tuple, List, Dict

from PIL import ImageTk

from app.buttons_and_texts import ButtonsAndTexts
from app.data_model import DataModel, ViewModel
from app.letter_images_frame import MainLettersScreen, DuplicateLettersFrame
from app.letters_in_page_handler import LettersInPageHandler
from app.special_values import NUM_OF_LETTERS, ZERO_X_Y, UNKNOWN_KEY, PAD_X
from app.tools import points_accumulate, points_sub, box_lines_from_center, two_points_min, two_points_max, \
    box_margin_from_box_shape


class MainWindow:
    def __init__(
            self,
            data_model: DataModel,
            on_look_for_letter_callback: Callable,
            get_image_patch: Callable,
            list_of_buttons_and_indicators: List[Tuple[bool, Tuple]],
            menu_values: Dict[str, List],
            translation: Tuple = ZERO_X_Y,
            page_view_shape=(700, 400)
    ):
        self._view_model = ViewModel(data_model)
        self._on_look_for_letter_callback = on_look_for_letter_callback
        self._get_image_patch_callback = get_image_patch
        self._translation = translation
        self._page_max_view_shape = page_view_shape

        self._window = tk.Tk(className=' Mareh - One Time OCR')
        self._im = tk.PhotoImage('../Mareh OCR.ico')
        self._window.iconbitmap(self._im)

        self._menu_elem = tk.Menu(self._window)

        menu_values['Find Letters'].append(('Correlate Letter', self._correlate_letter))
        for elem_name, sub_elements in menu_values.items():
            elem_menu = tk.Menu(self._menu_elem, tearoff=0)
            for name, func in sub_elements:
                elem_menu.add_command(label=name, command=func)
            self._menu_elem.add_cascade(label=elem_name, menu=elem_menu)
        self._window.config(menu=self._menu_elem)

        self._top_bar = tk.Frame(self._window)
        self._top_bar.grid(row=0, column=0, sticky="nsew")

        self._pages_state_bar = tk.Frame(self._top_bar)
        self._pages_state_bar.pack(side=tk.LEFT, padx=PAD_X)

        self._separator = ttk.Separator(self._top_bar, orient='vertical')
        self._separator.pack(side=tk.LEFT, fill=tk.Y, padx=PAD_X)

        self._main_letters_bar = tk.Frame(self._top_bar)
        self._main_letters_bar.pack(side=tk.LEFT, padx=PAD_X)

        self._pages_control_bar = tk.Frame(self._pages_state_bar)
        self._pages_control_bar.grid(row=0, column=0, sticky="nsew")

        self._buttons_and_indicators = ButtonsAndTexts(self._pages_control_bar, list_of_buttons_and_indicators)
        self._is_page_ready = tk.BooleanVar()
        self._is_page_ready_button = tk.Checkbutton(
            self._pages_state_bar, text="is ready", variable=self._is_page_ready, command=self._set_page_state
        )
        self._is_page_ready_button.grid(row=1, column=0, sticky="nsew")

        self._data_frame = ttk.Notebook(self._window)
        self._text_frame = ttk.Frame(self._data_frame)
        self._duplicates_letters_frame = ttk.Frame(self._data_frame)
        self._data_frame.add(self._text_frame, text='Page')
        self._data_frame.add(self._duplicates_letters_frame, text='Letters')
        self._data_frame.grid(row=2, column=0, sticky="nsew")
        self._canvas = tk.Canvas(self._text_frame)
        self._canvas.pack(side=tk.LEFT)
        self._canvas_image = None

        self._canvas.bind("<Button-1>", self._on_mouse_press_left)
        self._canvas.bind("<Button-2>", self._on_mouse_press_wheel)
        self._canvas.bind("<Button-3>", self._on_mouse_press_right)

        self._main_letters_handler = LettersInPageHandler(
            self._view_model, self._run_gui_action, self._canvas, self._get_letter_patch, self._translator
        )

        self._main_letters_screen = MainLettersScreen(
            self._view_model, self._run_gui_action, self._get_letter_by_key, self._main_letters_bar
        )

        self._duplicates_letters_screen = DuplicateLettersFrame(
            self._view_model, self._run_gui_action, self._get_letter_patch, self._duplicates_letters_frame
        )

        self._view_model.data_model.page.attach(self._update_image)

    @staticmethod
    def get_folder(start_dir: Path, text: str):
        return Path(filedialog.askdirectory(initialdir=str(start_dir), title=text))

    def _set_page_state(self):
        is_ready = self._is_page_ready.get()
        self._view_model.data_model.set_page_state(is_ready)

    def _update_image(self, page):
        if self._canvas_image:
            self._canvas.delete(self._canvas_image)
        new_pil_image = self._view_model.data_model.pil_image
        self._translation = ZERO_X_Y
        current_page_view_shape = two_points_min(new_pil_image.size, self._page_max_view_shape)
        self._view_model.current_page_view_shape = current_page_view_shape

        tk_letter_image = ImageTk.PhotoImage(new_pil_image)
        self._canvas.image = tk_letter_image
        self._canvas_image = self._canvas.create_image(0, 0, image=tk_letter_image, anchor=tk.NW)
        self._canvas.config(width=current_page_view_shape[0], height=current_page_view_shape[1])

        purple_box_lines_x_y = box_lines_from_center(box_margin_from_box_shape(current_page_view_shape), (4, 4))
        self._canvas.create_rectangle(*purple_box_lines_x_y, tags=('center',), outline='purple', width=5)
        self._view_model.data_model.reset_data()
        self._is_page_ready.set(self._view_model.data_model.is_page_ready_map[page])

    def _get_letter_by_key(self, key, scale=True):
        key_image = self._view_model.data_model.different_letters.data[key]
        return key_image[::2, ::2] if scale else key_image

    def _get_letter_patch(self, key):
        return self._get_image_patch_callback(self._view_model.data_model.pil_image, key)

    def _translator(self, location, inverse=False):
        return points_accumulate(location, self._translation, -1 if inverse else 1)

    def _run_gui_action(self, func, delay=0):
        return lambda *args, **kwargs: self._window.after(delay, func(*args, **kwargs))

    def _correlate_letter(self):
        window = tk.Tk(className='Correlate')
        label = tk.Label(window, text="chose num of correlations:")
        label.grid(row=0, column=0, sticky="nsew")

        num_duplicates_scale = tk.Scale(window, from_=1, to=200, orient=tk.HORIZONTAL)
        num_duplicates_scale.set(NUM_OF_LETTERS)
        num_duplicates_scale.grid(row=0, column=1, sticky="nsew")

        def func():
            val = num_duplicates_scale.get()
            window.destroy()
            self._on_look_for_letter(val)
        button_run = tk.Button(window, text="Run", command=func)
        button_run.grid(row=1, column=0, sticky="nsew")
        button_close = tk.Button(window, text="Cancel", command=window.destroy)
        button_close.grid(row=1, column=1, sticky="nsew")

    def _on_look_for_letter(self, num_duplicates):
        letter_key = self._view_model.current_chosen_letter.data
        instances_locations_by_letters = self._view_model.data_model.instances_locations_by_letters.data
        if letter_key in instances_locations_by_letters and instances_locations_by_letters[letter_key]:
            letter = next(iter((instances_locations_by_letters[letter_key])))
            self._on_look_for_letter_callback(letter_key, letter, num_duplicates)

    def _on_mouse_press_left(self, event):
        location = self._translator((event.x, event.y))
        if not self._view_model.current_chosen_letter.data:
            self._main_letters_handler.add_main_letter(UNKNOWN_KEY)
            self._view_model.current_chosen_letter.data = UNKNOWN_KEY

        self._main_letters_handler.add_dup_letter(self._view_model.current_chosen_letter.data, location)

    def _on_mouse_press_right(self, event):
        location = self._translator((event.x, event.y))
        self._main_letters_handler.add_main_letter(location + (self._view_model.data_model.page.data,))

    # TODO: support removing from screen?
    # def _on_mouse_press_delete(self, event):
        # instances_locations_by_letters = self._view_model.data_model.instances_locations_by_letters.data
        # for letter_location in list(instances_locations_by_letters.keys()):
        #     if are_points_close(letter_location, location):
        #         instances_locations_by_letters.pop(letter_location)
        # self._view_model.data_model.instances_locations_by_letters.data = instances_locations_by_letters

    def _on_mouse_press_wheel(self, event):
        page_size = self._view_model.data_model.pil_image.size
        point = (event.x, event.y)

        canvas_center = box_margin_from_box_shape(self._view_model.current_page_view_shape)
        new_translation = self._translator(points_sub(point, canvas_center))

        max_moves = points_sub(page_size, self._view_model.current_page_view_shape)
        new_translation = two_points_max(new_translation, ZERO_X_Y)
        new_translation = two_points_min(new_translation, max_moves)

        if new_translation != self._translation:
            diff = self._translator(new_translation, inverse=True)
            self._translation = new_translation
            self._canvas.move(self._canvas_image, *[-axis for axis in diff])
            self._view_model.data_model.reset_data()

    def run(self):
        self._window.mainloop()
