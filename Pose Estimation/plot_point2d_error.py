import matplotlib.pyplot as plt
import numpy as np

ours = [3.450055636724387, 3.3733760912698414, 3.293562202380952, 3.1374163690476196, 3.072929761904762, 3.0450249999999994, 2.997525]
small = [3.677507278138528, 3.5990663690476192, 3.5304441468253973, 3.4521872023809523, 3.3970309523809523, 3.3620666666666668, 3.29275]
nano = [3.9213682178932183, 3.865784126984127, 3.7767007936507935, 3.686520238095238, 3.6107791666666667, 3.5435291666666666, 3.476379166666667]
Chen = 2.667058330658729

fig = plt.figure()
ax = fig.subplots()
x = [i for i in range(7)]
ax.plot(nano, label='Ours-nano')
ax.scatter(x, nano)
ax.plot(small, label='Ours-small')
ax.scatter(x, small)
ax.plot(ours, label='Ours')
ax.scatter(x, ours)
ax.axhline(Chen, linestyle='--', label='Chen', color='black')
ax.set_ylabel('Mean Square Error [pixel]')
ax.set_xlabel('Top k')
plt.legend()
plt.xticks(x, [11-i for i in range(7)])
fig.savefig(r'C:\Users\Administrator\Desktop\python\local_python\Pose Estimation\zhanghaopeng\plot_py\topk_MeanSquareError.pdf')
fig.show()