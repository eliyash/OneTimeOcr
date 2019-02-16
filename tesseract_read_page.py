import pytesseract
from PIL import Image, ImageDraw
# If you don't have tesseract executable in your PATH, include the following:
from pytesseract import Output

from paths import Locations
from utils import make_dir

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


class Letter:
    def __init__(self, name: str, bounding_box: Square):
        self.name = name
        self.bounding_box = bounding_box


# Simple image to string
# print(pytesseract.image_to_string(Image.open(r"C:\Users\eli\Dropbox\Workspace\AI\mnist\text.bmp")))
#
# # French text image to string
# print(pytesseract.image_to_string(Image.open('test-european.jpg'), lang='fra'))


# Get bounding box estimates
img = Image.open(Locations.PAGE_TO_READ_PATH)
# img = img.filter(ImageFilter.FIND_EDGES)
# img = img.convert('L')
# img = img.point(lambda x: 0 if x < 180 else 255, '1')
# img.show()
word_data = pytesseract.image_to_data(img, output_type=Output.DICT, lang='heb')

boxes_of_words = []
for top, height, left, width in zip(word_data['top'], word_data['height'], word_data['left'], word_data['width']):
    boxes_of_words.append(Square(top, height, left, width))

letter_data = pytesseract.image_to_boxes(img, output_type=Output.DICT, lang='heb')


img_width, img_height = img.size
boxes_of_letters = []

letters_boxes = zip(
    letter_data['char'],
    letter_data['top'],
    letter_data['bottom'],
    letter_data['left'],
    letter_data['right']
)

index = 0
for char, top, bottom, left, right in letters_boxes:
    height = top - bottom
    top = img_height - top
    left = left
    width = right - left
    try:
        letter_image = img.copy()
        letter_image = letter_image.crop((left, top, left+width, top+height))
        if not char.isalpha():
            char = ord(char)
        file_path = "{}\\letter{}\\".format(Locations.LETTERS_PATH, char)
        make_dir(file_path)
        letter_image.save('{}{}.png'.format(file_path, index), "PNG")

        boxes_of_letters.append(Letter(char, Square(top, height, left, width)))
        index += 1
    except:
        print("error in letter: {}".format(letter_data['char'][index]))
draw = ImageDraw.Draw(img)
all_boxes = boxes_of_words + [letter.bounding_box for letter in boxes_of_letters]
for box in all_boxes:
    draw.line(box.get_surrounding_line(), fill=128)
del draw
img.show()

# # Get information about orientation and script detection
# print(pytesseract.image_to_osd(img))
#
