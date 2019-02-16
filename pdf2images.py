# from pdf2image import convert_from_path
# pages = convert_from_path('test_file.pdf', 500)
#
# for index, page in enumerate(pages):
#     page.save('./book/page_{}.jpg'.format(index), 'JPEG')
# import os, subprocess
#
# pdf_dir = r"C:\yourPDFfolder"
# os.chdir(pdf_dir)
#
# pdftoppm_path = r"C:\Program Files (x86)\Poppler\poppler-0.68.0\bin\pdftoppm.exe"
#
# for pdf_file in os.listdir(pdf_dir):
#
#     if pdf_file.endswith(".pdf"):
#
#         subprocess.Popen('"%s" -jpeg %s out' % (pdftoppm_path, pdf_file))
# Extract jpg's from pdf's. Quick and dirty.
import PyPDF2

from PIL import Image

import sys
from os import path
import warnings
warnings.filterwarnings("ignore")

number = 0

def recurse(page, xObject):
    global number

    xObject = xObject['/Resources']['/XObject'].getObject()

    for obj in xObject:

        if xObject[obj]['/Subtype'] == '/Image':
            size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
            data = xObject[obj]._data
            if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
                mode = "RGB"
            else:
                mode = "P"

            imagename = "%s - p. %s - %s"%(abspath[:-4], p, obj[1:])

            if xObject[obj]['/Filter'] == '/FlateDecode':
                img = Image.frombytes(mode, size, data)
                img.save(imagename + ".png")
                number += 1
            elif xObject[obj]['/Filter'] == '/DCTDecode':
                img = open(imagename + ".jpg", "wb")
                img.write(data)
                img.close()
                number += 1
            elif xObject[obj]['/Filter'] == '/JPXDecode':
                img = open(imagename + ".jp2", "wb")
                img.write(data)
                img.close()
                number += 1
        else:
            print(number)
            recurse(page, xObject[obj])



try:
    filename = 'test_file.pdf'
    pages = range(1,50)
    abspath = path.abspath(filename)
except BaseException:
    print('Usage :\nPDF_extract_images file.pdf page1 page2 page3 …')
    sys.exit()


file = PyPDF2.PdfFileReader(open(filename, "rb"))

for p in pages:
    page0 = file.getPage(p-1)
    recurse(p, page0)

print('%s extracted images'% number)