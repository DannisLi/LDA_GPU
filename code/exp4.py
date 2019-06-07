# coding:utf8

'''
thread_num nips
'''

import os
import matplotlib.pyplot as plt

# os.system('nvcc main.cu -w')


thread_num_list = [16, 32, 64, 128, 256]
gpu_perp_list, gpu_time_list = [],[]

for thread_num in thread_num_list:
    gpu_perp,gpu_time = 0,0
    for i in range(3):
        gpu_res = os.popen('./a.out nips 4 1 60 %d 5 3' % thread_num)
        gpu_res = gpu_res.readlines()
        gpu_perp += float(gpu_res[0].strip()) / 5.
        gpu_time += int(gpu_res[1].strip()) / 5.
    
    print(thread_num, gpu_perp, gpu_time)
    gpu_perp_list.append(gpu_perp)
    gpu_time_list.append(gpu_time)
    

fig = plt.figure(figsize=(24,12))
ax = fig.add_subplot(111)
ln1 = ax.plot(thread_num_list, gpu_perp_list, color='b', marker='o', label='log likelihood')
ax2 = ax.twinx()
ln2 = ax2.plot(thread_num_list, gpu_time_list, color='r', marker='o', label='used time')

fig.legend()

ax.set_xlabel('thread number')
ax.set_ylabel('log likelihood')
ax2.set_ylabel('used time (s)')
plt.savefig('nips_thread_num.png')
plt.close()
