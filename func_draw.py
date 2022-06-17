import matplotlib.pyplot as plt
import numpy as np

def plot_(xlabel, ylabel, history):
    plt.plot(np.arange(len(history)), history)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()

def plot_multi(xlabel, history_list):
    for history in history_list:
        plt.plot(np.arange(len(history)), history)
    plt.xlabel(xlabel)
    plt.show()

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def draw_multi(l_ylabel, l_data, l_color, r_ylabel, r_data, r_color, filename=None, title=None):
    fig, ax1 = plt.subplots()
    if title: plt.title(title)
    plt.xlabel('episode')
    ax2 = ax1.twinx()

    ax1.set_ylabel(l_ylabel, color=l_color)
    ax1.plot(np.arange(len(l_data)), l_data, color=l_color, alpha=0.3)
    lns1 = ax1.plot(np.arange(len(l_data)), smooth_curve(l_data), label=l_ylabel, color=l_color, alpha=1)
    ax1.tick_params(axis='y', labelcolor=l_color)

    ax2.set_ylabel(r_ylabel, color=r_color)
    ax2.plot(np.arange(len(r_data)), r_data, color=r_color, alpha=0.3)
    lns2 = ax2.plot(np.arange(len(r_data)), smooth_curve(r_data), label=r_ylabel, color=r_color, alpha=1)
    ax2.tick_params(axis='y', labelcolor=r_color)

    # ax1.legend(loc='upper left', bbox_to_anchor=(0.5, -0.15), ncol=3, fancybox=False, shadow=False)
    # ax2.legend(loc='upper right', bbox_to_anchor=(0.5, -0.15), ncol=3, fancybox=False, shadow=False)

    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    fig.tight_layout()
    if filename: plt.savefig('./fig/paper8_'+filename+'.png', dpi=500, bbox_inches='tight')
    plt.show()