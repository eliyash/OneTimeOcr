import random
from pathlib import Path
from typing import Dict, Optional
from PIL import Image
from app.observers import Subject
from app.tools import UNKNOWN_KEY, get_values_to_add_and_remove


class DataModel:
    def __init__(self, image_paths: Path):
        self.images_paths = [path for path in image_paths.iterdir()]
        self.instances_locations_per_image = [dict() for _ in self.images_paths]
        self.is_page_ready_map = [False for _ in self.images_paths]

        self.different_letters = Subject(dict())
        self.instances_locations_by_letters = Subject(dict())

        self.page = Subject()
        self.current_page = None  # type: Optional[int]
        self.pil_image = None
        self.image_path = None

        self.page.attach(self.set_page)
        self.different_letters.attach(self._on_different_letters)

    def _on_different_letters(self, different_letters: Dict):
        self.instances_locations_per_image[self.current_page] = self.instances_locations_by_letters.data
        for page, instances_locations in enumerate(self.instances_locations_per_image):
            to_remove, to_add = get_values_to_add_and_remove(instances_locations, different_letters)
            instances_locations.update({key: set() for key in to_add})
            if to_remove:
                self.is_page_ready_map[page] = False
                removed_values = [instances_locations.pop(key) for key in to_remove]
                unknown_values = instances_locations[UNKNOWN_KEY] if UNKNOWN_KEY in instances_locations else set()
                unknown_values.union({location for elem_in_pop in removed_values for location in elem_in_pop})
                instances_locations[UNKNOWN_KEY] = unknown_values
        self.instances_locations_by_letters.data = self.instances_locations_per_image[self.current_page]

    @property
    def num_of_pages(self):
        return len(self.images_paths)

    def set_page_state(self, value):
        self.is_page_ready_map[self.current_page] = value

    def set_page(self, index: int):
        if self.current_page:
            self.instances_locations_per_image[self.current_page] = self.instances_locations_by_letters.data
        self.image_path = self.images_paths[index]
        self.pil_image = Image.open(str(self.image_path))
        self.instances_locations_by_letters.data = self.instances_locations_per_image[index]
        self.current_page = index

    def reset_data(self):
        random.seed(0)
        current_data = self.instances_locations_by_letters.data
        self.instances_locations_by_letters.data = {k: set() for k in current_data.keys()}
        self.instances_locations_by_letters.data = current_data


class ViewModel:
    def __init__(self, data_modal: DataModel):
        self.data_model = data_modal

        self.current_main_letters = Subject(set())
        self.current_chosen_letter = Subject()
        self.current_location_duplicates = Subject(set())

        self.current_chosen_letter.attach(self.set_new_chosen_letter)
        self.data_model.instances_locations_by_letters.attach(self.handle_main_letters_change)
        self.data_model.instances_locations_by_letters.attach(self.set_current_location_duplicates)

        self.map_keys_by_widgets = {}

    def handle_main_letters_change(self, new_instances_locations_by_letters: Dict):
        current_chosen_letter = self.current_chosen_letter.data
        new_main_letters = set(new_instances_locations_by_letters.keys())
        letters_to_add = new_main_letters - self.current_main_letters.data
        if letters_to_add:
            self.current_chosen_letter.data = next(iter(letters_to_add))
        elif current_chosen_letter in new_main_letters:
            pass
        elif new_main_letters:
            self.current_chosen_letter.data = next(iter(new_main_letters))
        else:
            self.current_chosen_letter.data = None

        self.current_main_letters.data = new_main_letters

    def set_current_location_duplicates(self, new_instances_locations_by_letters: Dict):
        main_letter = self.current_chosen_letter.data
        if main_letter in new_instances_locations_by_letters:
            new_current_location_duplicates = new_instances_locations_by_letters[main_letter]
            if self.current_location_duplicates.data != new_current_location_duplicates:
                self.current_location_duplicates.data = new_current_location_duplicates

    def set_new_chosen_letter(self, new_main_letter):
        instances_locations_by_letters = self.data_model.instances_locations_by_letters.data
        if new_main_letter in instances_locations_by_letters:
            new_current_location_duplicates = instances_locations_by_letters[new_main_letter]
        else:
            new_current_location_duplicates = []
        self.current_location_duplicates.data = new_current_location_duplicates
