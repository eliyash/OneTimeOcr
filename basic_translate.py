import pytesseract
from PIL import Image
import io
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

img = Image.open(r"C:\Users\eli\Dropbox\Workspace\AI\mnist\text.bmp")

text = pytesseract.image_to_string(img, output_type=Output.STRING, lang='heb')

with io.open("Output.txt", "w", encoding="utf-8") as text_file:
    text_file.write(text)
