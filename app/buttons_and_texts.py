import tkinter as tk
from typing import Callable, Tuple, List, Union
from app.observers import Subject


class ButtonsAndTexts:
    def __init__(self, bar, buttons_and_indicators: List[Tuple[bool, Tuple]]):
        self._bar = bar
        self._buttons_and_indicators = [
            self._create_label(is_button, params) for is_button, params in buttons_and_indicators
        ]

    def _create_label(self, is_button: bool, params: Tuple[str, Union[Callable, Subject]]):
        if is_button:
            text, func = params
            return self._create_button(text, func)
        else:
            text, notifier = params
            return self._create_indicator(text, notifier)

    def _create_button(self, text: str, func: Callable):
        button = tk.Button(self._bar, text=text, command=func)
        button.pack(side=tk.LEFT)
        return button

    def _create_indicator(self, text: str, notifier: Subject):
        text_var = tk.StringVar()
        label = tk.Label(self._bar, textvariable=text_var)
        label.pack(side=tk.LEFT)

        def func(new_val):
            text_var.set('{}: {}'.format(text, new_val))

        notifier.attach(func)
        return label, text_var
