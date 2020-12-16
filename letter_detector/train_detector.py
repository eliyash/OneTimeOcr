import json
from pathlib import Path

import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
import time

from app.paths import TRAIN_DATA_PATH, IMAGES_PATH
from app.tools import get_device
from letter_detector.dataset import CustomDataset
from letter_detector.loss import Loss
from letter_detector.model import EAST

BASIC_EAST_MODEL = '../networks/pretrained/east_vgg16.pth'


def train(images_path: Path, gt_data_path: Path, train_portion: float, networks_path: Path,
          batch_size, lr, num_workers, number_of_epochs, minimal_improvement_for_save, start_net=None):
    gt_path = gt_data_path / 'pages'
    device = get_device()
    data_set = CustomDataset(images_path, gt_path, duplicate_pages=batch_size)

    size = len(data_set)
    if size < 2 * batch_size:
        train_set, test_set = data_set, data_set
    else:
        train_size = int(size * train_portion)
        train_set, test_set = torch.utils.data.random_split(data_set, [train_size, size-train_size])

    train_loader = data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    test_loader = data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )

    file_num = len(train_set)

    criterion = Loss()

    model = EAST()
    if start_net:
        model.load_state_dict(torch.load(start_net))  # added

    data_parallel = False
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        data_parallel = True
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[number_of_epochs // 2], gamma=0.1)

    best_training_loss = None
    for epoch in range(number_of_epochs):
        model.train()
        scheduler.step(epoch)
        epoch_loss = 0
        epoch_time = time.time()
        for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
            start_time = time.time()
            loss = calc_loss_on_net(criterion, device, gt_geo, gt_score, ignored_map, img, model)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(
                epoch + 1, number_of_epochs, i + 1, int(file_num / batch_size), time.time() - start_time, loss.item())
            )

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (img, gt_score, gt_geo, ignored_map) in enumerate(test_loader):
                loss = calc_loss_on_net(criterion, device, gt_geo, gt_score, ignored_map, img, model)
                test_loss += loss.item()

        print('test_loss is {:.8f}, epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(
            test_loss / int(file_num / batch_size), epoch_loss / int(file_num / batch_size), time.time() - epoch_time)
        )
        print(time.asctime(time.localtime(time.time())))
        if best_training_loss is None or test_loss < (best_training_loss - minimal_improvement_for_save):
            best_training_loss = test_loss
            new_net_name = 'model_epoch_{}.pth'.format(epoch + 1)
            state_dict = model.module.state_dict() if data_parallel else model.state_dict()
            torch.save(state_dict, str(networks_path / new_net_name))
            print('new net saved, name: {}, loss: {}'.format(new_net_name, test_loss))
            with open(str(networks_path / 'state.json'), 'w') as state_file:
                json.dump({'loss': best_training_loss}, state_file)
        print('=' * 80)


def calc_loss_on_net(criterion, device, gt_geo, gt_score, ignored_map, img, model):
    img, gt_score, gt_geo, ignored_map = \
        img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
    pred_score, pred_geo = model(img)
    loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
    return loss


def run_train(data_set):
    torch.manual_seed(1)
    train_portion = 0.5
    start_net = BASIC_EAST_MODEL
    batch_size = 4
    lr = 1e-3
    num_workers = 4
    number_of_epochs = 20
    minimal_improve_to_save = 0
    network_path = Path('../networks') / 'detector' / time.strftime("train_%Y%m%d-%H%M%S")
    Path(network_path).mkdir(parents=True)
    train(
        IMAGES_PATH, TRAIN_DATA_PATH / data_set, train_portion, network_path, batch_size, lr,
        num_workers, number_of_epochs, minimal_improve_to_save, start_net
    )


def main():
    run_train('dataset_20201209-145735')


if __name__ == '__main__':
    main()
