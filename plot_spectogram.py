import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from src.Dataset_handlers.MAFATDataset import MAFATDataset
from src.Classifier.RadarClassifier import PreProcessLayer
from src.utils import load_classifier_config, create_spectogram

compare_to_pre_process_layer = False

name = 'test'
config = load_classifier_config()
dataset = MAFATDataset(config, name)
seg_id_list = [6760,
               6682,
               6994,
               6891,
               6712,
               6743,
               6686,
               6935,
               6796,
               6932,
               6737,
               6666,
               6890,
               6982,
               6909,
               6893,
               6791,
               6719,
               6978,
               6664,
               6945,
               6668]

for seg_id in seg_id_list:
    iq_data, label = dataset[int(np.where(dataset.segment_ids==seg_id)[0])]

    spectogram_from_numpy = create_spectogram(iq_data.numpy())

    color_map = 'viridis'
    if compare_to_pre_process_layer:
        ppl = PreProcessLayer('real', True, False)
        spectogram_array = ppl.forward(iq_data)
        spectogram = spectogram_array[0,0,:,:].numpy()
        fig, axes = plt.subplots(1,3)
        ax = sns.heatmap(spectogram, cmap=color_map, ax=axes[0])
        ax.set_title('Torch, label={}'.format(label))
        ax = sns.heatmap(spectogram_from_numpy, cmap=color_map, ax=axes[1])
        ax.set_title('Numpy, label={}'.format(label))
        ax = sns.heatmap(spectogram_from_numpy - spectogram, cmap=color_map, ax=axes[2])
        ax.set_title('diff')
    else:
        plt.clf()
        ax = sns.heatmap(spectogram_from_numpy, cmap=color_map)
        ax.set_title('Numpy, label={}'.format(label))
        ax.figure.savefig(os.path.join('test_debug', str(seg_id) + '.png'))
