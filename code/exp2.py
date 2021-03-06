# coding:utf8

'''
n_epoch nips
'''

import os
import matplotlib.pyplot as plt

# os.system('nvcc main.cu -w')



n_epoch_list = list(range(10, 101, 10))
cpu_perp_list, cpu_time_list = [],[]
gpu_perp_list, gpu_time_list = [],[]

for n_epoch in n_epoch_list:
    gpu_perp,cpu_perp = 0,0
    gpu_time,cpu_time = 0,0
    for i in range(5):
        gpu_res = os.popen('./a.out nips 8 1 %d 256 5 1' % n_epoch)
        cpu_res = os.popen('./a.out nips 8 0 %d' % n_epoch)
    
        gpu_res = gpu_res.readlines()
        cpu_res = cpu_res.readlines()
    
        gpu_perp += float(gpu_res[0].strip()) / 5.
        cpu_perp += float(cpu_res[0].strip()) / 5.

        gpu_time += int(gpu_res[1].strip()) / 5.
        cpu_time += int(cpu_res[1].strip()) / 5.
    
    # time = int(res[1].strip())
    print(n_epoch, gpu_perp, cpu_perp)
    cpu_perp_list.append(cpu_perp)
    gpu_perp_list.append(gpu_perp)

    cpu_time_list.append(cpu_time)
    gpu_time_list.append(gpu_time)
    # time_list.append(time)

plt.figure(figsize=(24,12))
plt.plot(n_epoch_list, cpu_perp_list, label='cpu', color='b', marker='o')
plt.plot(n_epoch_list, gpu_perp_list, label='gpu', color='r', marker='o')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('log likelihood')
plt.savefig('nips_perp.png')
plt.close()

plt.clf()

plt.figure(figsize=(24,12))
plt.plot(n_epoch_list, cpu_time_list, label='cpu', color='b', marker='o')
plt.plot(n_epoch_list, gpu_time_list, label='gpu', color='r', marker='o')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('used time (s)')
plt.savefig('nips_time.png')
plt.close()