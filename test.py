import torch

from data import TongRen
from model import model
import tqdm
import numpy as np
from torch.nn import functional
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    for idx in range(50, 1000, 50):
        data = TongRen(train=False)

        network = model()
        mp = torch.load("save/{}.pt".format(idx))
        network.load_state_dict(mp['state_dict'])
        network = network.cuda()
        total = 0
        ok = 0
        for i in range(data.__len__()):
            ret, img_type = data.__getitem__(i)
            cnt = torch.zeros([4])
            for j in range(5):
                img = ret[j].cuda()
                img = torch.unsqueeze(img, dim=0)
                res = network.forward(img)
                res = torch.softmax(res, dim=1)
                res = torch.argmax(res).cpu()
                cnt[res] += 1
            fin = int(torch.argmax(cnt))
            total += 1
            if fin == img_type:
                ok += 1
        print(idx, ok, total, ok / total)
