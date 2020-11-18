from pathlib import Path

import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
import time

from letter_detector.dataset import CustomDataset
from letter_detector.loss import Loss
from letter_detector.model import EAST


def train(train_data_path: Path, test_data_path: Path, pths_path: Path,
          batch_size, lr, num_workers, epoch_iter, minimal_improvement_for_save, start_net=None):
    train_set = CustomDataset(str(train_data_path / 'image'), str(train_data_path / 'letters'))
    train_loader = data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    test_set = CustomDataset(str(test_data_path / 'image'), str(test_data_path / 'letters'))
    test_loader = data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )

    file_num = len(train_set)

    criterion = Loss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    if start_net:
        model.load_state_dict(torch.load(start_net))  # added

    data_parallel = False
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        data_parallel = True
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epoch_iter // 2], gamma=0.1)

    best_training_loss = None
    for epoch in range(epoch_iter):
        model.train()
        scheduler.step(epoch)
        epoch_loss = 0
        epoch_time = time.time()
        for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
            start_time = time.time()
            img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(
                device)
            pred_score, pred_geo = model(img)
            loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)

            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(
                epoch + 1, epoch_iter, i + 1, int(file_num / batch_size), time.time() - start_time, loss.item())
            )

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (img, gt_score, gt_geo, ignored_map) in enumerate(test_loader):
                img, gt_score, gt_geo, ignored_map = \
                    img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
                pred_score, pred_geo = model(img)
                loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
                test_loss += loss.item()

        print('test_loss is {:.8f}, epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(
            test_loss / int(file_num / batch_size), epoch_loss / int(file_num / batch_size), time.time() - epoch_time)
        )
        print(time.asctime(time.localtime(time.time())))
        if not best_training_loss or test_loss < (best_training_loss - minimal_improvement_for_save):
            best_training_loss = test_loss
            new_net_name = 'model_epoch_{}.pth'.format(epoch + 1)
            state_dict = model.module.state_dict() if data_parallel else model.state_dict()
            torch.save(state_dict, str(pths_path / new_net_name))
            print('new net saved, name: {}, loss: {}'.format(new_net_name, test_loss))
        print('=' * 80)


def main():
    time_str = time.strftime("%Y%m%d-%H%M%S")
    start_net = '../networks/pretrained/east_vgg16.pth'

    train_data_path = Path(r'C:\Workspace\MyOCR\EAST\eli east gez\train')
    test_data_path = Path(r'C:\Workspace\MyOCR\EAST\eli east gez\test')
    pths_path = Path('./nets') / time_str
    Path(pths_path).mkdir(parents=True)
    batch_size = 4
    lr = 1e-3
    num_workers = 4
    epoch_iter = 1000
    minimal_improvement_for_save = 0
    train(train_data_path, test_data_path, pths_path,
          batch_size, lr, num_workers, epoch_iter, minimal_improvement_for_save, start_net)


if __name__ == '__main__':
    main()
