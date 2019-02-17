class Locations:
    PAGE_LETTERS_DIRECTORY = r'.\data\letters by page\\'
    PAGES_DIRECTORY = r'.\data\books\\'
    NETWORK_DIRECTORY = r'.\data\networks\\'
    TESSERACT_EXEC = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
    TRAINING_LETTERS = r'.\data\letters by page\training letters'
    TESSERACT_RESULT_FOLDER = r'C:\Workspace\MyOCR\data\tesseract result\\'
    PAGE_TO_READ = 'page_3'
    NETWORK_NAME = 'last_net.pt'

    PAGE_TO_READ_PATH = PAGES_DIRECTORY + PAGE_TO_READ + ".png"
    LETTERS_PATH = PAGE_LETTERS_DIRECTORY + PAGE_TO_READ
    NETWORK_PATH = NETWORK_DIRECTORY + NETWORK_NAME
