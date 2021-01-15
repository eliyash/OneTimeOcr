from pathlib import Path

DATA_ROOT = Path(r"../../data")
IMAGES_PATH = DATA_ROOT / 'images'
LETTERS_PATH = DATA_ROOT / 'annotations'
TRAIN_DATA_PATH = DATA_ROOT / 'train_dataset'

NETWORK_ROOT = DATA_ROOT / 'networks'
IDENTIFIER_NETS_PATH = NETWORK_ROOT / 'identifier'
DETECTOR_NETS_PATH = NETWORK_ROOT / 'detector'
