from __future__ import print_function
import torch
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from mnist_like_net import Net
from paths import Locations


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = f.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {}'.format(epoch))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += f.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss


def main():
    torch.manual_seed(1)

    device = "cuda"

    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.Resize((100, 100)),
        # transforms.Resize((110, 110)),
        # transforms.RandomCrop(100, 100),
        transforms.RandomRotation(10),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image_net_data = ImageFolder(Locations.TRAINING_LETTERS, transform=transform)

    train_loader = torch.utils.data.DataLoader(image_net_data,
                                               batch_size=16,
                                               shuffle=True,
                                               num_workers=1)

    test_loader = torch.utils.data.DataLoader(image_net_data,
                                              batch_size=16,
                                              shuffle=True,
                                              num_workers=1)

    model = Net(len(image_net_data.classes)).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(1, 3):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    torch.save(model, Locations.NETWORK_PATH)


if __name__ == '__main__':
    main()
