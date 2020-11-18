import torch
import numpy as np
from PIL import Image
from letter_detector.detect import detect
from letter_detector.model import EAST


def detect_letters(img_path, model_path="../networks/letter_detector/model_epoch_137.pth"):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = EAST().to(device)
	model.load_state_dict(torch.load(model_path))
	model.eval()
	img = Image.open(img_path)
	
	boxes = detect(img, model, device)

	locations = set()
	for box in np.array(boxes):
		locations.add((int(np.mean(box[0:-1:2])), int(np.mean(box[1::2]))))

	return locations


detect_letters(r"C:\Workspace\MyOCR\EAST\test_pages\test_gez good - Copy.jpg")