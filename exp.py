

# ['cpu' or 'gpu']['']

data = {'cpu': {}, 'gpu': {}}
data['cpu']

参数：
设备
线程数
同步频率
总轮次

结果：
对数似然
用时

cpu_data：迭代轮数，性能类型
gpu_data: 迭代轮数，线程数，同步次数，性能类型

gpu_kos_data[20][256][5]['perp'] = -3604697.277395
gpu_kos_data[20][256][5]['time'] = 2

gpu_kos_data[50][256][5]['perp'] = -3533415.313281
gpu_kos_data[50][256][5]['time'] = 4

gpu_kos_data[100][256][5]['perp'] = -3526299.480126
gpu_kos_data[100][256][5]['time'] = 8

gpu_kos_data[150][256][5]['perp'] = -3528619.095046
gpu_kos_data[150][256][5]['time'] = 12

gpu_kos_data[200][256][5]['perp'] = -3525679.747450
gpu_kos_data[200][256][5]['time'] = 16



cpu_kos_data[20]['perp'] = -3564298.175238
cpu_kos_data[20]['time'] = 1

cpu_kos_data[40]['perp'] = -3562655.044735
cpu_kos_data[40]['time'] = 1

cpu_kos_data[50]['perp'] = -3563526.117362
cpu_kos_data[50]['time'] = 2

cpu_kos_data[100]['perp'] = -3560721.374783
cpu_kos_data[100]['time'] = 3

cpu_kos_data[150]['perp'] = -3559786.639224
cpu_kos_data[150]['time'] = 6

cpu_kos_data[200]['perp'] = -3558645.409743
cpu_kos_data[200]['time'] = 6



Please input corpus name: my
Please input number of topics: 2
n_epoch 100
CPU
perp: -47850.973916
used time: 1 s
GPU 同步 10 线程 32
perp: -47923.560024
used time: 150 s
GPU 同步 10 线程 32
perp: -47923.560024
used time: 150 s
GPU 同步 5 线程 32
perp: -47859.321472
used time: 147 s
GPU 同步 1 线程 32
perp: -47857.426468
used time: 141 s
GPU 同步 2 线程 32
perp: -47851.303134
used time: 144 s
GPU 同步 2 线程 64
perp: -47851.105344
used time: 96 s
GPU 同步 2 线程 128
perp: -47854.884812
used time: 53 s
GPU 同步 2 线程 256
perp: -47848.136999
used time: 34 s
GPU 同步 2 线程 512
perp: -47852.889986
used time: 23 s
GPU 同步 2 线程 1024
perp: -47858.992154
used time: 15 s
GPU 同步 5 线程 1024
perp: -47851.728088
used time: 16 s
GPU 同步 10 线程 1024
perp: -47921.862716
used time: 15 s
GPU 同步 20 线程 1024
perp: -47990.402273
used time: 16 s

Please input corpus name: kos
Please input number of topics: 4
n_epoch 100
CPU
perp: -3654810.850082
used time: 3 s
GPU 同步 2 线程 1024
perp: -3655279.328965
used time: 1261 s
GPU 同步 5 线程 1024
perp: -3655661.186767
used time: 764 s
GPU 同步 10 线程 1024
perp: -3670141.541600
used time: 590 s
GPU 同步 10 线程 1024
perp: -3671339.719296
used time: 186 s


