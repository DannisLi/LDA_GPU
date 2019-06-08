# coding:utf8

'''
thread_num kos
'''

import os
import matplotlib.pyplot as plt

# os.system('nvcc main.cu -w')


thread_num_list = [16, 32, 64, 128, 256]
gpu_perp_list, gpu_time_list = [],[]

for thread_num in thread_num_list:
    gpu_perp,gpu_time = 0,0
    for i in range(5):
        gpu_res = os.popen('./a.out kos 4 1 60 %d 5 0' % thread_num)
        gpu_res = gpu_res.readlines()
        gpu_perp += float(gpu_res[0].strip()) / 5.
        gpu_time += int(gpu_res[1].strip()) / 5.
    
    print(thread_num, gpu_perp, gpu_time)
    gpu_perp_list.append(gpu_perp)
    gpu_time_list.append(gpu_time)
    

fig = plt.figure(figsize=(24,12))
ax = fig.add_subplot(111)
ax.plot(thread_num_list, gpu_perp_list, color='b', marker='o', label='log likelihood')
ax2 = ax.twinx()
ax2.plot(thread_num_list, gpu_time_list, color='r', marker='o', label='used time')

fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)

ax.set_xlabel('thread number')
ax.set_ylabel('log likelihood')
ax2.set_ylabel('used time (s)')
plt.savefig('kos_thread_num.png')
plt.close()
