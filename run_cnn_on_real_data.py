from __future__ import print_function

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import transforms

from mnist_like_net import Net
from paths import Locations


def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    # image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image


def main():
    model = torch.load(Locations.NETWORK_PATH)  # type: Net
    model.eval()

    data_transforms = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    # image = Image.open(Locations.IMAGE_TO_TEST)
    image = image_loader(data_transforms, Locations.IMAGE_TO_TEST)

    var_image = Variable(torch.Tensor(image))
    print(model(var_image))


if __name__ == '__main__':
    main()
