import json
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def load_classifier_config():
    with open('src/Classifier/config.json') as config_file:
        config = json.load(config_file)
    return config


def load_seperator_config():
    with open('src/Separator/config.json') as config_file:
        config = json.load(config_file)
    return config


def box_print(msg):
    print("=" * max(len(msg), 100))
    print(msg)
    print("=" * max(len(msg), 100))


def create_spectogram(x, shift_map:bool = True, zero_pad:bool = True, scale:bool = True):
    FT_AX = 0
    n_fast_time = x.shape[FT_AX]
    x = x - np.mean(x, axis=FT_AX, keepdims=True)
    x = np.multiply(x, np.hamming(n_fast_time).reshape((n_fast_time, -1)))
    x = np.fft.fft(x, n=256 if zero_pad else 128, axis=FT_AX)
    x = np.log(np.abs(x) + 1)
    x = np.fft.fftshift(x, FT_AX) if shift_map else x
    if scale:
        x = x - np.min(x.flatten())
        x = x / np.max(x.flatten())
    return x


def plot_spectogram(x, shift_map:bool = True, zero_pad:bool = True, scale:bool = True, ax=None, show_plot=True):
    if ax is None:
        plt.clf()
    spectogram = create_spectogram(x, shift_map, zero_pad, scale)
    color_map = 'viridis'
    ax = sns.heatmap(spectogram, cmap=color_map, ax=ax)
    if show_plot:
        plt.show()
    return ax