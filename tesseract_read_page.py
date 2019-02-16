import errno
import os

from PIL import Image, ImageDraw, ImageFilter
import pytesseract

# If you don't have tesseract executable in your PATH, include the following:
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'


def make_dir(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

# Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

# Simple image to string
# print(pytesseract.image_to_string(Image.open(r"C:\Users\eli\Dropbox\Workspace\AI\mnist\text.bmp")))
#
# # French text image to string
# print(pytesseract.image_to_string(Image.open('test-european.jpg'), lang='fra'))

# Get bounding box estimates
page_num = 3
img = Image.open(r".\text{}.png".format(page_num))
# img = img.filter(ImageFilter.FIND_EDGES)
# img = img.convert('L')
# img = img.point(lambda x: 0 if x < 180 else 255, '1')
# img.show()
word_data = pytesseract.image_to_data(img, output_type=Output.DICT, lang='heb')
height_list = word_data['height']
left_list = word_data['left']
top_list = word_data['top']
width_list = word_data['width']

boxes_of_words = []
for index in range(len(top_list)):
    width, left, top, height = width_list[index], left_list[index], top_list[index], height_list[index]
    boxes_of_words.append(
        [(left, top), (left, top + height), (left + width, top + height), (left + width, top), (left, top)])

letter_data = pytesseract.image_to_boxes(img, output_type=Output.DICT, lang='heb')

right_list = letter_data['right']
left_list = letter_data['left']
top_list = letter_data['top']
bottom_list = letter_data['bottom']

width, height = img.size
boxes_of_letters = []

for index in range(len(top_list)):
    try:
        left, right, top, bottom = left_list[index], right_list[index], top_list[index], bottom_list[index]
        letter_image = img.copy()
        letter_image = letter_image.crop((left, height-top, right, height-bottom))
        char = letter_data['char'][index]
        if not char.isalpha():
            char = ord(char)
        file_path = ".\\page_{}\\letter{}\\".format(page_num, char)
        make_dir(file_path)
        letter_image.save('{}{}.png'.format(file_path, index), "PNG")

        boxes_of_letters.append(
            [(left, height-top), (left, height-bottom), (right, height-bottom), (right, height-top), (left, height-top)])
    except:
        print("error in letter: {}".format(letter_data['char'][index]))
draw = ImageDraw.Draw(img)
all_boxes = boxes_of_letters + boxes_of_words
for box in all_boxes:
    draw.line(box, fill=128)
del draw
img.show()

# # Get information about orientation and script detection
# print(pytesseract.image_to_osd(img))
#
