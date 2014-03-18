from collections import OrderedDict
import pdb

import matplotlib.pyplot as plt
import numpy as np

gpu_times = OrderedDict()
gpu_times['default'] = 0.5166
gpu_times[(1, 1)] = 0.3911
gpu_times[(2, 1)] = 0.2609
gpu_times[(4, 1)] = 0.2145
gpu_times[(8, 1)] = 0.1098
gpu_times[(16, 1)] = 0.2234
gpu_times[(1, 16)] = 0.02729
gpu_times[(1, 8)] = 0.05856
gpu_times[(1, 4)] = 0.1111
gpu_times[(1, 2)] = 0.2083
gpu_times[(2, 8)] = 0.03433
gpu_times[(8, 2)] = 0.05926
gpu_times[(4, 4)] = 0.05409

gpu_labels = [str(label) for label in gpu_times.keys()]
gpu_labels.insert(0, '')

cpu_times = OrderedDict()
cpu_times['default'] = 0.07326
cpu_times[(1, 1)] = 0.1427
cpu_times[(2, 1)] = 0.1442
cpu_times[(4, 1)] = 0.07309
cpu_labels = [str(label) for label in cpu_times.keys()]

serial_time = 0.1173

fig = plt.figure()
plt.xticks(rotation=70)
ax = fig.add_subplot(111)

n_gpu = len(gpu_times)
n_cpu = len(cpu_times)
gpu = ax.bar(range(n_gpu), gpu_times.values(), align='center',
        label='PyOpenCL on Intel HD 4000', color='green')
cpu = ax.bar(range(n_gpu, n_gpu + n_cpu), cpu_times.values(), align='center',
        label='PyOpenCL on Intel i5-3427U @ 1.80GHz', color='blue')
serial = ax.bar(n_gpu + n_cpu, serial_time, align='center',
        label='Serial NumPy on Intel i5-3427U @ 1.80GHz', color='red')

ax.set_xlim([0, n_cpu + n_gpu])
ax.set_xticks(np.arange(-1, n_cpu + n_gpu + 2, 1))
ax.set_xticklabels(gpu_labels + cpu_labels + ['N/A'])

ax.set_xlabel('Local block size')
ax.set_ylabel('Time per 512x512 multiplication (s)')

ax.legend()
fig.savefig('benchmark.png', bbox_inches='tight', dpi=600)
fig.savefig('benchmark.pdf', bbox_inches='tight')
