# coding:utf8

import os
import matplotlib.pyplot as plt

os.system('nvcc main.cu -w')

name = 'kos'

n_epoch_list = list(range(10, 201, 10))
cpu_perp_list, cpu_time_list = [],[]
gpu_perp_list, gpu_time_list = [],[]

for n_epoch in n_epoch_list:
    gpu_res = os.popen('./a.out kos 4 1 %d 256 5' % n_epoch)
    cpu_res = os.popen('./a.out kos 4 0 %d' % n_epoch)
    
    gpu_res = gpu_res.readlines()
    cpu_res = cpu_res.readlines()
    
    gpu_perp = float(gpu_res[0].strip())
    cpu_perp = float(cpu_res[0].strip())
    
    # time = int(res[1].strip())
    print(n_epoch, gpu_perp, cpu_perp)
    cpu_perp_list.append(cpu_perp)
    gpu_perp_list.append(gpu_perp)
    # time_list.append(time)


plt.plot(n_epoch_list, cpu_perp_list, label='cpu', c='b')
plt.plot(n_epoch_list, gpu_perp_list, label='gpu', c='r')
plt.legend()
plt.savefig('perp.png')

