# coding:utf8

import os
import matplotlib.pyplot as plt

os.system('nvcc main.cu -w')

name = 'kos'

perp_list, time_list = [],[]

for n_epoch in range(10, 201, 10):
    # res = os.popen('./a.out kos 4 1 %d 256 5' % n_epoch)
    res = os.popen('./a.out kos 4 0 %d' % n_epoch)
    res = res.readlines()
    # print(res)
    perp = float(res[0].strip())
    time = int(res[1].strip())
    print(n_epoch, perp, time)
    perp_list.append(perp)
    time_list.append(time)


plt.plot(list(range(10,201,10)), perp_list)
plt.savefig('perp.png')

