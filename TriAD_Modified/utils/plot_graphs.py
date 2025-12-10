# Created by: Thuan Anh Bui (2025)
# Description: Enabled displaying graphs and charts for results.

import numpy as np
import matplotlib.pyplot as plt

import torch
print(torch.__version__)
print(torch.version.cuda)

# ---------------------------------------------------------
# 1) BAR CHARTS: Highest vs Lowest for F1, Recall, Precision
# ---------------------------------------------------------

# Colors
light_blue = "#8ecae6"
light_red  = "#ff7f7f"

# Label modes (edit if you have different names)
label_modes = ['ANY', 'BOTH', 'OVERLOAD\nCONDITION', 'TRANSFORMER\nFAULT']

# F1 scores
f1_high = [0.201, 0.0112, 0.171, 0.063]   # highest F1 in each label mode
f1_low  = [0.153, 0.0073, 0.157, 0.029]   # lowest F1 in each label mode

# Recall
recall_high = [0.474, 0.6626, 0.486, 0.540]  # highest Recall in each label mode
recall_low  = [0.194, 0.3272, 0.301, 0.361]  # lowest Recall in each label mode

# Precision
precision_high = [0.523, 0.5828, 0.516, 0.506]  # highest Precision in each label mode
precision_low  = [0.517, 0.4827, 0.514, 0.494]  # lowest Precision in each label mode

x = np.arange(len(label_modes))  # positions for modes
width = 0.35                     # width of each bar

def plot_high_low(metric_name, high_values, low_values):
    plt.figure()
    plt.bar(x - width/2, high_values, width, label='Highest', color=light_blue)
    plt.bar(x + width/2, low_values, width, label='Lowest', color=light_red)

    plt.xticks(x, label_modes)
    plt.ylabel(metric_name)
    plt.title(f'{metric_name}: Highest vs Lowest per Label Mode')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

# Plot for each metric
plot_high_low("F1 Score", f1_high, f1_low)
plot_high_low("Recall", recall_high, recall_low)
plot_high_low("Precision", precision_high, precision_low)

# ---------------------------------------------------------
# 2) cycle = 6.0, stride = 3, Recall vs Uni/Multi (ALL MODES)
# ---------------------------------------------------------

modes_6_3 = label_modes                 # use all modes
x_6_3 = np.arange(len(modes_6_3))
width_6_3 = 0.35

# Recall values for cycle=6.0, stride=3
recall_6_3_uni = [0.300, 0.4464, 0.301, 0.361]
recall_6_3_multi = [0.302, 0.5029, 0.313, 0.365]

plt.figure()
plt.bar(x_6_3 - width_6_3/2, recall_6_3_uni,   width_6_3, label='Univariate')
plt.bar(x_6_3 + width_6_3/2, recall_6_3_multi, width_6_3, label='Multivariate')

plt.xticks(x_6_3, modes_6_3)
plt.ylabel('Recall')
plt.title('Recall (cycle=6.0, stride=3)')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 4) cycle = 12.0, stride = 4, Recall vs Uni/Multi (ALL MODES)
# ---------------------------------------------------------

modes_12_4 = label_modes             # reuse same modes list
x_12_4 = np.arange(len(modes_12_4))
width_12_4 = 0.35

# Recall values for cycle=12.0, stride=4 (order must match label_modes)
recall_12_4_uni   = [0.474, 0.6626, 0.474, 0.510]
recall_12_4_multi = [0.466, 0.634, 0.486, 0.540]

plt.figure()
plt.bar(x_12_4 - width_12_4/2, recall_12_4_uni, width_12_4, label='Univariate')
plt.bar(x_12_4 + width_12_4/2, recall_12_4_multi, width_12_4, label='Multivariate')

plt.xticks(x_12_4, modes_12_4)
plt.ylabel('Recall')
plt.title('Recall (cycle=12.0, stride=4)')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()
