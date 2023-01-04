import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

orientation_error_bydet = 'orientation_error_bydet.csv'
orientation_error_byreg = 'orientation_error_byreg.csv'
position_error_bydet = 'position_error_bydet.csv'
position_error_byreg = 'position_error_byreg.csv'
orientation_error_byheatmap = 'orientation_error_byheatmap.csv'
position_error_byheatmap = 'position_error_byheatmap.csv'

orientation_error_bydet = np.array(pd.read_csv(orientation_error_bydet, header=None), dtype='float')
orientation_error_byreg = np.array(pd.read_csv(orientation_error_byreg, header=None), dtype='float')
position_error_bydet = np.array(pd.read_csv(position_error_bydet, header=None), dtype='float')
position_error_byreg = np.array(pd.read_csv(position_error_byreg, header=None), dtype='float')
orientation_error_byheatmap = np.array(pd.read_csv(orientation_error_byheatmap, header=None), dtype='float')
position_error_byheatmap = np.array(pd.read_csv(position_error_byheatmap, header=None), dtype='float')

bar = 100
std_ratio = 0.1
orientation_error_byreg_new = []
orientation_error_bydet_new = []
orientation_error_byheatmap_new = []
for i in range(orientation_error_byreg.shape[0]//bar):
    mean_orientation_error_byreg = np.mean(orientation_error_byreg[i*bar:(i+1)*bar, 0])
    max_orientation_error_byreg = np.max(orientation_error_byreg[i * bar:(i + 1) * bar, 0])
    min_orientation_error_byreg = np.min(orientation_error_byreg[i * bar:(i + 1) * bar, 0])
    std_orientation_error_byreg = np.std(orientation_error_byreg[i * bar:(i + 1) * bar, 0]) * std_ratio
    mean_orientation_error_bydet = np.mean(orientation_error_bydet[i * bar:(i + 1) * bar, 0])
    max_orientation_error_bydet = np.max(orientation_error_bydet[i * bar:(i + 1) * bar, 0])
    min_orientation_error_bydet = np.min(orientation_error_bydet[i * bar:(i + 1) * bar, 0])
    std_orientation_error_bydet = np.std(orientation_error_bydet[i * bar:(i + 1) * bar, 0]) * std_ratio
    mean_orientation_error_byheatmap = np.mean(orientation_error_byheatmap[i * bar:(i + 1) * bar, 0])
    max_orientation_error_byheatmap = np.max(orientation_error_byheatmap[i * bar:(i + 1) * bar, 0])
    min_orientation_error_byheatmap = np.min(orientation_error_byheatmap[i * bar:(i + 1) * bar, 0])
    std_orientation_error_byheatmap = np.std(orientation_error_byheatmap[i * bar:(i + 1) * bar, 0]) * std_ratio
    mean_distance = np.mean(np.mean(orientation_error_bydet[i * bar:(i + 1) * bar, 1]))
    orientation_error_byreg_new.append([mean_orientation_error_byreg, std_orientation_error_byreg, max_orientation_error_byreg, min_orientation_error_byreg, mean_distance])
    orientation_error_bydet_new.append([mean_orientation_error_bydet, std_orientation_error_bydet, max_orientation_error_bydet, min_orientation_error_bydet, mean_distance])
    orientation_error_byheatmap_new.append(
        [mean_orientation_error_byheatmap, std_orientation_error_byheatmap, max_orientation_error_byheatmap,
         min_orientation_error_byheatmap, mean_distance])
orientation_error_byreg_new = np.array(orientation_error_byreg_new)
orientation_error_bydet_new = np.array(orientation_error_bydet_new)
orientation_error_byheatmap_new = np.array(orientation_error_byheatmap_new)
x_tick = orientation_error_byreg_new[:, -1]
plt.plot(x_tick, orientation_error_byreg_new[:, 0], c='b', label='Tae')
plt.fill_between(x_tick, orientation_error_byreg_new[:, 0] + orientation_error_byreg_new[:, 1],
                 orientation_error_byreg_new[:, 0] - orientation_error_byreg_new[:, 1], color='b', alpha=0.2)
plt.plot(x_tick, orientation_error_byheatmap_new[:, 0], c='g', label='Chen')
plt.fill_between(x_tick, orientation_error_byheatmap_new[:, 0] + orientation_error_byheatmap_new[:, 1],
                 orientation_error_byheatmap_new[:, 0] - orientation_error_byheatmap_new[:, 1], color='g', alpha=0.2)
plt.plot(x_tick, orientation_error_bydet_new[:, 0], c='r', label='Ours')
plt.fill_between(x_tick, orientation_error_bydet_new[:, 0] + orientation_error_bydet_new[:, 1],
                 orientation_error_bydet_new[:, 0] - orientation_error_bydet_new[:, 1], color='r', alpha=0.2)
# plt.xticks(x_tick, orientation_error_bydet_new[:, -1])
plt.ylabel('Orientation Error [deg]')
plt.xlabel('Relative Distance [m]')
plt.legend()
plt.savefig('comp_orientation_error_dif_dis.pdf')
plt.show()

bar = 100
std_ratio = 0.1
position_error_byreg_new = []
position_error_bydet_new = []
position_error_byheatmap_new = []
for i in range(orientation_error_byreg.shape[0]//bar):
    mean_position_error_byreg = np.mean(position_error_byreg[i*bar:(i+1)*bar, 0])
    max_position_error_byreg = np.max(position_error_byreg[i * bar:(i + 1) * bar, 0])
    min_position_error_byreg = np.min(position_error_byreg[i * bar:(i + 1) * bar, 0])
    std_position_error_byreg = np.std(position_error_byreg[i * bar:(i + 1) * bar, 0]) * std_ratio
    mean_position_error_bydet = np.mean(position_error_bydet[i * bar:(i + 1) * bar, 0])
    max_position_error_bydet = np.max(position_error_bydet[i * bar:(i + 1) * bar, 0])
    min_position_error_bydet = np.min(position_error_bydet[i * bar:(i + 1) * bar, 0])
    std_position_error_bydet = np.std(position_error_bydet[i * bar:(i + 1) * bar, 0]) * std_ratio
    mean_distance = np.mean(np.mean(position_error_bydet[i * bar:(i + 1) * bar, 1]))
    mean_position_error_byheatmap = np.mean(position_error_byheatmap[i * bar:(i + 1) * bar, 0])
    max_position_error_byheatmap = np.max(position_error_byheatmap[i * bar:(i + 1) * bar, 0])
    min_position_error_byheatmap = np.min(position_error_byheatmap[i * bar:(i + 1) * bar, 0])
    std_position_error_byheatmap = np.std(position_error_byheatmap[i * bar:(i + 1) * bar, 0]) * std_ratio
    position_error_byreg_new.append([mean_position_error_byreg, std_position_error_byreg, max_position_error_byreg, min_position_error_byreg, mean_distance])
    position_error_bydet_new.append([mean_position_error_bydet, std_position_error_bydet, max_position_error_bydet, min_position_error_bydet, mean_distance])
    position_error_byheatmap_new.append(
        [mean_position_error_byheatmap, std_position_error_byheatmap, max_position_error_byheatmap,
         min_position_error_byheatmap, mean_distance])
position_error_byreg_new = np.array(position_error_byreg_new)
position_error_bydet_new = np.array(position_error_bydet_new)
position_error_byheatmap_new = np.array(position_error_byheatmap_new)
x_tick = position_error_byreg_new[:, -1]
plt.plot(x_tick, position_error_byreg_new[:, 0], c='b', label='Tae')
plt.fill_between(x_tick, position_error_byreg_new[:, 0] + position_error_byreg_new[:, 1],
                 position_error_byreg_new[:, 0] - position_error_byreg_new[:, 1], color='b', alpha=0.2)
plt.plot(x_tick, position_error_byheatmap_new[:, 0], c='g', label='Chen')
plt.fill_between(x_tick, position_error_byheatmap_new[:, 0] + position_error_byheatmap_new[:, 1],
                 position_error_byheatmap_new[:, 0] - position_error_byheatmap_new[:, 1], color='g', alpha=0.2)
plt.plot(x_tick, position_error_bydet_new[:, 0], c='r', label='Ours')
plt.fill_between(x_tick, position_error_bydet_new[:, 0] + position_error_bydet_new[:, 1],
                 position_error_bydet_new[:, 0] - position_error_bydet_new[:, 1], color='r', alpha=0.2)
# plt.xticks(x_tick, orientation_error_bydet_new[:, -1])
plt.ylabel('Position Error [deg]')
plt.xlabel('Relative Distance [m]')
plt.legend()
plt.savefig('comp_position_error_dif_dis.pdf')
plt.show()

