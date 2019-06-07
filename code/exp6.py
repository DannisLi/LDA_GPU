# coding:utf8

'''
sync_epoch kos
'''

import os
import matplotlib.pyplot as plt

# os.system('nvcc main.cu -w')


sync_epoch_list = [2, 4, 6, 10, 15]
gpu_perp_list, gpu_time_list = [],[]

for sync_epoch in sync_epoch_list:
    gpu_perp,gpu_time = 0,0
    for i in range(5):
        gpu_res = os.popen('./a.out nips 4 1 60 256 %d 2' % sync_epoch)
        gpu_res = gpu_res.readlines()
        gpu_perp += float(gpu_res[0].strip()) / 5.
        gpu_time += int(gpu_res[1].strip()) / 5.
    
    print(sync_epoch, gpu_perp, gpu_time)
    gpu_perp_list.append(gpu_perp)
    gpu_time_list.append(gpu_time)
    

fig = plt.figure(figsize=(24,12))
ax = fig.add_subplot(111)
ax.plot(sync_epoch_list, gpu_perp_list, color='b', marker='o', label='log likelihood')
ax2 = ax.twinx()
ax2.plot(sync_epoch_list, gpu_time_list, color='r', marker='o', label='used time')

fig.legend()

ax.set_xlabel('epochs per sync')
ax.set_ylabel('log likelihood')
ax2.set_ylabel('used time (s)')
plt.savefig('nips_sync_epoch.png')
plt.close()
