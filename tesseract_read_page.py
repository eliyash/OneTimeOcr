from typing import List, Tuple

import pytesseract
from PIL import Image, ImageDraw
# If you don't have tesseract executable in your PATH, include the following:
from pytesseract import Output

from utils import make_dir
from paths import Locations

pytesseract.pytesseract.tesseract_cmd = Locations.TESSERACT_EXEC


class Square:
    def __init__(self, top: int, height: int, left: int, width: int):
        self.top = top
        self.height = height
        self.left = left
        self.width = width

    def get_surrounding_line(self):
        top, height, left, width = self.top, self.height, self.left, self.width
        return [(left, top), (left, top + height), (left + width, top + height), (left + width, top), (left, top)]

    def get_image_cropping(self):
        top, height, left, width = self.top, self.height, self.left, self.width
        return left, top, left + width, top + height

    def clone(self):
        return Square(self.top, self.height, self.left, self.width)

    def match_top_and_height(self, match_to: "Square"):
        self.top = match_to.top
        self.height = match_to.height

    # TODO: check logic
    @staticmethod
    def _do_lines_overlap(line1: Tuple[int, int], line2: Tuple[int, int]):
        if line2[0] <= line1[0] < line2[1]:
            return True
        if line1[0] <= line2[0] < line1[1]:
            return True

        return False

    # TODO: check logic
    @classmethod
    def do_squares_overlap(cls, square_1: "Square", square_2: "Square"):
        if not cls._do_lines_overlap(
            (square_1.left, square_1.left + square_1.width),
            (square_2.left, square_2.left + square_2.width)
        ):
            return False

        if not cls._do_lines_overlap(
            (square_1.top, square_1.top + square_1.height),
            (square_2.top, square_2.top + square_2.height)
        ):
            return False

        return True


class Letter:
    def __init__(self, name: str, bounding_box: Square):
        self.name = name
        self.bounding_box = bounding_box

    def clone(self):
        return Letter(self.name, self.bounding_box.clone())


class Word:
    def __init__(self, line: int, word_str: str, bounding_box: Square, letters: List[Letter] = None):
        self.line = line
        self.word_str = word_str
        self.bounding_box = bounding_box
        self.letters = letters
        if self.letters is None:
            self.letters = []

    def match_letters_to_word_boxes(self):
        for letter in self.letters:
            letter.bounding_box.match_top_and_height(self.bounding_box)

    def clone(self):
        return Word(self.line, self.word_str, self.bounding_box.clone(), [letter.clone() for letter in self.letters])


def generate_word_boxes(img):
    word_data = pytesseract.image_to_data(img, output_type=Output.DICT, lang='heb')
    # from utils import save_data, load_data
    # word_data = load_data(Locations.TESSERACT_RESULT_FOLDER + "word_data.json")
    # save_data(word_data, Locations.TESSERACT_RESULT_FOLDER + "word_data.json")

    boxes_from_tesseract = zip(
        word_data['top'],
        word_data['height'],
        word_data['left'],
        word_data['width'],
        word_data['line_num'],
        word_data['level'],
        word_data['text']
    )

    all_words = []
    line_words = []
    for top, height, left, width, line, level, text in boxes_from_tesseract:
        if level == 5:
            line_words.append(Word(line, text, Square(top, height, left, width)))
        if level == 4:
            line_words.reverse()
            all_words.extend(line_words)
            line_words = []
    line_words.reverse()
    all_words.extend(line_words)
    return all_words


def generate_letter_boxes(img):
    letter_data = pytesseract.image_to_boxes(img, output_type=Output.DICT, lang='heb')
    # from utils import save_data, load_data
    # letter_data = load_data(Locations.TESSERACT_RESULT_FOLDER + "letter_data.json")
    # save_data(letter_data, Locations.TESSERACT_RESULT_FOLDER + "letter_data.json")

    img_width, img_height = img.size

    letters_boxes = zip(
        letter_data['char'],
        letter_data['top'],
        letter_data['bottom'],
        letter_data['left'],
        letter_data['right']
    )

    boxes_of_letters = []
    for char, top, bottom, left, right in letters_boxes:
        height = top - bottom
        top = img_height - top
        left = left
        width = right - left
        boxes_of_letters.append(Letter(char, Square(top, height, left, width)))
    return boxes_of_letters


def save_letter_images(img, letters: List[Letter]):
    index = 0
    for letter in letters:
        char = letter.name
        try:
            letter_image = img.copy()
            letter_image = letter_image.crop(letter.bounding_box.get_image_cropping())
            if not char.isalpha():
                char = ord(char)
            file_path = r"{}\letter{}\\".format(Locations.LETTERS_PATH, char)
            make_dir(file_path)
            letter_image.save('{}{}.png'.format(file_path, index), "PNG")

            index += 1
        except Exception as e:
            print("error in letter: {}, e={}".format(char, e))


def match_letters_with_words(words: List[Word], letters: List[Letter]):
    last_letters = 0
    for word in words:
        last_letter_in_current_word = last_letters + len(word.word_str)
        word.letters = letters[last_letters:last_letter_in_current_word]
        last_letters = last_letter_in_current_word


def index_to_color(index):
    r = (index * 60) % 255
    g = (index * 30) % 255
    b = (index * 15) % 255
    return r, g, b, 255


def generate_image_with_boxes(orig_image: Image, words: List[Word]):
    new_image = orig_image.copy()
    draw = ImageDraw.Draw(new_image)
    index = 0
    word_letters = []
    for word in words:
        for letter in word.letters:
            color = (0, 0, 0, 255)
            if index:
                color = index_to_color(index)

                word_letter = letter.clone()

                draw.line(word_letter.bounding_box.get_surrounding_line(), fill=color)
                word_letters.append(word_letter)

            draw.line(word.bounding_box.get_surrounding_line(), fill=color)
        index += 1
    del draw
    return new_image


def main():
    orig_image = Image.open(Locations.PAGE_TO_READ_PATH)

    letters = generate_letter_boxes(orig_image)
    words = generate_word_boxes(orig_image)

    match_letters_with_words(words, letters)

    for word in words:
        word.match_letters_to_word_boxes()

    generate_image_with_boxes(orig_image, words).show()

    save_letter_images(orig_image, [letter for word in words for letter in word.letters])


if __name__ == '__main__':
    main()
