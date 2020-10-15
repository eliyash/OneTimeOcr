class Locations:
    PAGE_LETTERS_DIRECTORY = r'.\data\letters by page\\'
    PAGES_DIRECTORY = r'.\data\books\\'
    NETWORK_DIRECTORY = r'.\data\networks\\'
    TESSERACT_EXEC = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    TRAINING_LETTERS = r'.\data\letters by page\training letters'
    TESSERACT_RESULT_FOLDER = r'C:\Workspace\MyOCR\data\tesseract result\\'
    PAGE_TO_READ = 'page_gez.jpg'
    NETWORK_NAME = 'last_net.pt'
    IMAGE_TO_TEST = r'C:\Workspace\MyOCR\data\letters by page\page_3\letterא\20.png'

    PAGE_TO_READ_PATH = PAGES_DIRECTORY + PAGE_TO_READ
    LETTERS_PATH = PAGE_LETTERS_DIRECTORY + PAGE_TO_READ
    NETWORK_PATH = NETWORK_DIRECTORY + NETWORK_NAME
