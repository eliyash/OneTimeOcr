from app.tools import BOX_WIDTH_MARGIN, BOX_HEIGHT_MARGIN


class MarkerManager:
    def __init__(self, canvas, color, box_width_margin=BOX_WIDTH_MARGIN, box_height_margin=BOX_HEIGHT_MARGIN):
        self._instances_locations = set()
        self._canvas = canvas
        self._color = color
        self._box_width_margin = box_width_margin
        self._box_height_margin = box_height_margin

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

    def add_letter(self, location):
        self._instances_locations.add(location)
        self._add_a_box(location)

    def remove_letter(self, letter_location):
        self._instances_locations.remove(letter_location)
        self._remove_a_box(letter_location)

    def set_all_letters(self, letters):
        self.remove_all_letters()
        for letter in letters:
            self.add_letter(letter)

    def remove_all_letters(self):
        for letter in self._instances_locations:
            self._remove_a_box(letter)
