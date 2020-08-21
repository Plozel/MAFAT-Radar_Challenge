import os
import sys
import random
import torch
import yaml
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint

from torch.utils.data import random_split

from asteroid.metrics import get_metrics
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.models import ConvTasNet
from asteroid.utils import tensors_to_device
from asteroid.models import save_publishable

path_components = __file__.split(os.path.sep)
Code_folder_path = os.path.join(*path_components[:path_components.index('src')+1])
sys.path.append(os.path.sep + Code_folder_path if os.name == 'posix' else Code_folder_path)

from utils import plot_spectogram
from Dataset_handlers.SeparationDataset import SeparationDataset


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str,
                    help='One of `enh_single`, `enh_both`, '
                         '`sep_clean` or `sep_noisy`', default='sep_clean')
parser.add_argument('--test_dir', type=str,
                    help='Test directory including the json files', default='src/Separator/exp')
parser.add_argument('--use_gpu', type=int, default=1,
                    help='Whether to use the GPU for model execution')
parser.add_argument('--exp_dir', default='src/Separator/exp',
                    help='Experiment root')
parser.add_argument('--n_save_ex', type=int, default=10,
                    help='Number of audio examples to save, -1 means all')

compute_metrics = ['si_sdr', 'sdr', 'sir', 'sar', 'stoi']

def to_complex(x):
    x = x.reshape(-1,32,128,2)
    x = x[:,:,:,0] + x[:,:,:,1]*1j
    x = np.swapaxes(x, 1, 2)
    return x


def main(conf):
    model_path = os.path.join(conf['exp_dir'], 'best_model.pth')
    model = ConvTasNet.from_pretrained(model_path)
    # Handle device placement
    if conf['use_gpu']:
        model.cuda()
    model_device = next(model.parameters()).device

    # get data for evaluation - this should change in the future to work on real test data the was not used for training
    dataset = SeparationDataset(combination_list_path=os.path.join(conf['exp_dir'], 'combination_list.pkl'))
    n_val = int(len(dataset) * conf['train_conf']['data']['fraction_of_examples_to_use_for_validation'])
    train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val])  # noqa

    # test_set = val_set
    test_set = train_set
    # Used to reorder sources only
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')

    # Randomly choose the indexes of sentences to save.
    ex_save_dir = os.path.join(conf['exp_dir'], 'examples/')
    if conf['n_save_ex'] == -1:
        conf['n_save_ex'] = len(test_set)
    save_idx = random.sample(range(len(test_set)), conf['n_save_ex'])
    # series_list = []
    torch.no_grad().__enter__()
    for idx in tqdm(range(len(test_set))):
        # Forward the network on the mixture.
        mix, sources = tensors_to_device(test_set[idx], device=model_device)
        est_sources = model(mix[None, None])
        loss, reordered_sources = loss_func(est_sources, sources[None],
                                            return_est=True)  # noqa
        mix_np = to_complex(mix[None].cpu().data.numpy())
        sources_np = to_complex(sources.cpu().data.numpy())
        est_sources_np = to_complex(reordered_sources.squeeze(0).cpu().data.numpy())
        # utt_metrics = get_metrics(mix_np, sources_np, est_sources_np,
        #                           sample_rate=conf['sample_rate'],
        #                           metrics_list=compute_metrics)
        # utt_metrics['mix_path'] = test_set.mix[idx][0]
        # series_list.append(pd.Series(utt_metrics))

        # Save some examples in a folder. Wav files and metrics as text.
        if idx in save_idx:
            local_save_dir = os.path.join(ex_save_dir, 'ex_{}/'.format(idx))
            os.makedirs(local_save_dir, exist_ok=True)
            iq_data = mix_np[0]
            ax = plot_spectogram(iq_data, scale=False, show_plot=False)
            ax.figure.savefig(local_save_dir + 'mixture.png')
            # Loop over the sources and estimates
            for src_idx, src in enumerate(sources_np):
                iq_data = src
                ax = plot_spectogram(iq_data, scale=False, show_plot=False)
                ax.figure.savefig(local_save_dir + "s{}.png".format(src_idx+1))
            for src_idx, est_src in enumerate(est_sources_np):
                # est_src *= np.max(np.abs(mix_np))/np.max(np.abs(est_src))
                iq_data = np.reshape(est_src, (32,128)).T
                ax = plot_spectogram(iq_data, scale=False, show_plot=False)
                ax.figure.savefig(local_save_dir + "s{}_estimate.png".format(src_idx+1))
            # Write local metrics to the example folder.
            # with open(local_save_dir + 'metrics.json', 'w') as f:
            #     json.dump(utt_metrics, f, indent=0)

    # Save all metrics to the experiment folder.
    # all_metrics_df = pd.DataFrame(series_list)
    # all_metrics_df.to_csv(os.path.join(conf['exp_dir'], 'all_metrics.csv'))

    # # Print and save summary metrics
    # final_results = {}
    # for metric_name in compute_metrics:
    #     input_metric_name = 'input_' + metric_name
    #     ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
    #     final_results[metric_name] = all_metrics_df[metric_name].mean()
    #     final_results[metric_name + '_imp'] = ldf.mean()
    # print('Overall metrics :')
    # pprint(final_results)
    # with open(os.path.join(conf['exp_dir'], 'final_metrics.json'), 'w') as f:
    #     json.dump(final_results, f, indent=0)

    # model_dict = torch.load(model_path, map_location='cpu')
    # os.makedirs(os.path.join(conf['exp_dir'], 'publish_dir'), exist_ok=True)
    # publishable = save_publishable(
    #     os.path.join(conf['exp_dir'], 'publish_dir'), model_dict,
    #     metrics=final_results, train_conf=train_conf
    # )


if __name__ == '__main__':
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    # Load training config
    conf_path = os.path.join(args.exp_dir, 'conf.yml')
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)
    arg_dic['train_conf'] = train_conf

    if args.task != arg_dic['train_conf']['data']['task']:
        print("Warning : the task used to test is different than "
              "the one from training, be sure this is what you want.")

    main(arg_dic)