import os
import random
import shutil

full = r"E:\tongren\full"
train = r"E:\tongren\train"
test = r"E:\tongren\test"
dir = os.listdir(full)
for i in dir:
    v = os.listdir(os.path.join(full, i))
    random.shuffle(v)
    test_size = len(v) // 10
    v1 = v[:test_size]
    v2 = v[test_size:]
    os.makedirs(os.path.join(test, i))
    for j in v1:
        src = os.path.join(full, i, j)
        dst = os.path.join(test, i, j)
        shutil.copyfile(src, dst)
    os.makedirs(os.path.join(train, i))
    for j in v2:
        src = os.path.join(full, i, j)
        dst = os.path.join(train, i, j)
        shutil.copyfile(src, dst)
