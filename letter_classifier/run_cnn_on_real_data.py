from __future__ import print_function

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import transforms

from letter_classifier.default_locations import DEFAULT_IMAGES_FOLDER, DEFAULT_NETWORK_PATH
from letter_classifier.mnist_like_net import Net


def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    # image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image


def run_cnn_net(network_path, image_to_test):
    model = torch.load(network_path)  # type: Net
    model.eval()

    data_transforms = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    # image = Image.open(Locations.IMAGE_TO_TEST)
    image = image_loader(data_transforms, image_to_test)

    var_image = Variable(torch.Tensor(image))
    output = model(var_image)
    print(output)

    sm = torch.nn.Softmax()
    probabilities = sm(output)
    print(probabilities)  # Converted to probabilities


def main():
    run_cnn_net(DEFAULT_NETWORK_PATH, DEFAULT_IMAGES_FOLDER)


if __name__ == '__main__':
    main()
