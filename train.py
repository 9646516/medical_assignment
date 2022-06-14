import torch

from data import TongRen
from model import model
import tqdm
import numpy as np
from torch.nn import functional
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count
        return self


def get_training_lr(epoch, train_epoch, begin_lr, min_lr):
    ret = np.power(1.0 - epoch / float(train_epoch + 1), 0.9) * begin_lr
    return max(min_lr, ret)


save_dir = "/home/rinne/tongren/save"


def ohem(result, gt):
    loss = torch.nn.functional.cross_entropy(result, gt, reduction="none")
    loss, _ = torch.topk(loss, loss.shape[0] // 2)
    loss = torch.mean(loss)
    return loss


def focal(result, gt):
    loss = torch.nn.functional.cross_entropy(result, gt, reduction="none")
    p = F.softmax(result, dim=1) * gt
    p = torch.sum(p, dim=1)
    loss = loss * torch.square(1 - p)
    loss = torch.mean(loss)
    return loss


def simple(result, gt):
    loss = torch.nn.functional.cross_entropy(result, gt, reduction="none")
    loss = torch.mean(loss)
    return loss


if __name__ == '__main__':
    data = TongRen(train=True)
    dataLoader = torch.utils.data.DataLoader(
        dataset=data,
        batch_size=20,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=24,
    )
    network = model().cuda()
    opt = torch.optim.SGD(network.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    l1_meter = AverageMeter()
    log = SummaryWriter(flush_secs=10)
    base_iters = 0
    for epoch in range(1000):
        lr = get_training_lr(epoch, 1000, 1e-3, 1e-6)
        for param_group in opt.param_groups:
            param_group['lr'] = lr
        with tqdm.tqdm(dataLoader, unit="step") as tepoch:
            tepoch.set_description("Epoch {}".format(epoch))
            for idx, (img, gt) in enumerate(tepoch):
                base_iters += 1
                img = img.cuda()
                gt = gt.cuda()
                result = network.forward(img)
                loss = simple(result, gt)
                l1_meter.update(float(loss))
                log.add_scalar("loss", float(loss), base_iters)
                opt.zero_grad()
                loss.backward()
                opt.step()

                tepoch.set_postfix({
                    "loss": l1_meter.avg,
                })
        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': network.state_dict(),
            }, "{}/{}.pt".format(save_dir, epoch))
