import os
import numpy as np
import pandas as pd
from scipy import signal as sg

import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch_lightning import seed_everything
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.metrics import Accuracy, F1, Recall, Precision


from kornia.losses import FocalLoss

from ..utils import load_classifier_config
from ..Dataset_handlers.MAFATDataset import MAFATDataset, MAFATDatasetAugmented


# sets seeds for numpy, torch, etc...
# must do for DDP to work well
seed_everything(123)

class PreProcessLayer(LightningModule):
    def __init__(self, mode: str = 'real', zero_pad: bool = True, run_on_gpu: bool = True):
        super().__init__()
        self.output_complex = mode == 'complex'
        self.zero_pad = zero_pad
        self.n_fft = 256 if self.zero_pad else 128
        sampling_windows = np.concatenate((np.hamming(128).reshape(128, -1),
                                           np.blackman(128).reshape(128, -1),
                                           sg.windows.tukey(128).reshape(128, -1),
                                           sg.windows.parzen(128, np.pi).reshape(128, -1)), axis=1).T
        sampling_windows = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(sampling_windows), 0), 3).type(torch.cfloat)
        self.sampling_windows = sampling_windows.cuda() if run_on_gpu else sampling_windows
        self.n_win = sampling_windows.shape[1]
        
    def forward(self, x):
        x = x.view((-1, 128, 32))
        # Subtract the mean:
        x = x - torch.mean(x.real, dim=1, keepdim=True) - torch.mean(x.imag, dim=1, keepdim=True)*1j
        x = x.view((-1, 1, 128, 32))  # (batch_size, unitary dimension for sampling_windows, fast_time, slow_time)
        # Perform an FFT:
        x = x.expand(-1, self.n_win, 128, 32) * self.sampling_windows
        x = torch.view_as_real(x.permute(0, 1, 3, 2))
        x = nn.functional.pad(x, (0, 0, 0, 128)) if self.zero_pad else x
        x = torch.fft(x, 1)
        x = torch.view_as_complex(x).permute(0, 1, 3, 2)
        # Perform an FFT shift to bring the zero relative velocity to the center:
        x = torch.roll(x, int(0.5 * self.n_fft), dims=2)
        # Compute norms in log scale:
        if self.output_complex:
            x_norms = torch.log(torch.abs(x) + 1)
            x1 = torch.median(x_norms.view((-1, self.n_win, self.n_fft * 32)), dim=2, keepdim=True)[0] - 1
            x = torch.sign(x.real) * torch.log(torch.abs(x.real) + 1) + \
                    torch.sign(x.imag) * torch.log(torch.abs(x.imag) + 1)*1j
            x2 = torch.max(x_norms.view((-1, self.n_win, self.n_fft * 32)), dim=2, keepdim=True)[0]
            y1 = 0.05 / x2
            y2 = 1 / x2
            a1 = y1 / x1
            a2 = ((y2 - y1) / (x2 - x1))
            b2 = y2 - a2 * x2
            low_norm_multiplier = a1.view((-1, self.n_win, 1, 1)) * x
            high_norm_multiplier = a2.view((-1, self.n_win, 1, 1)) * x + b2.view((-1,self.n_win,1,1))
            multiplier_3d = torch.where(x_norms > x1.view((-1, self.n_win, 1, 1)), high_norm_multiplier, low_norm_multiplier)
            x = x * multiplier_3d
            return x.real, x.imag
        else:
            x = torch.log(torch.abs(x) + 1)
            thresh = torch.median(x.view((-1, self.n_win, self.n_fft * 32)), dim=2, keepdim=True)[0] - 1
            # Scale the data:
            x = x - thresh.view((-1,self.n_win,1,1))
            x = torch.where(x < 0, torch.zeros_like(x), x)
            max_val = torch.max(x.view((-1, self.n_win, self.n_fft * 32)), dim=2, keepdim=True)[0]
            x = x / max_val.view((-1,self.n_win,1,1))
            return x


class Inception(LightningModule):
    def __init__(self, in_channels, nof1x1, nof3x3_1, nof3x3_out, nof5x5_1, nof5x5_out, pool_planes):
        super(Inception, self).__init__()

        # 1x1 conv branch
        self.b1x1 = nn.Sequential(
            nn.Conv2d(in_channels, nof1x1, kernel_size=1),
            nn.BatchNorm2d(nof1x1),
            nn.ReLU(),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b1x3 = nn.Sequential(
            nn.Conv2d(in_channels, nof3x3_1, kernel_size=1),
            nn.BatchNorm2d(nof3x3_1),
            nn.ReLU(),
            nn.Conv2d(nof3x3_1, nof3x3_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(nof3x3_out),
            nn.ReLU(),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b1x5 = nn.Sequential(
            nn.Conv2d(in_channels, nof5x5_1, kernel_size=1),
            nn.BatchNorm2d(nof5x5_1),
            nn.ReLU(),
            nn.Conv2d(nof5x5_1, nof5x5_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(nof5x5_out),
            nn.ReLU(),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b3x1 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(),
        )

    def forward(self, a):
        b1 = self.b1x1.forward(a)
        b2 = self.b1x3.forward(a)
        b3 = self.b1x5.forward(a)
        b4 = self.b3x1.forward(a)
        return torch.cat([b1, b2, b3, b4], 1)


class ConvNet(LightningModule):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.pre_process_layer = PreProcessLayer(mode='real', zero_pad=True, run_on_gpu=True).requires_grad_(False)

        self.first_layer = nn.Sequential(
            nn.Conv2d(self.pre_process_layer.n_win, 30, kernel_size=3, padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU(),
        )

        self.a3 = Inception(30, 10,  4, 12, 4, 8, 8)
        self.b3 = Inception(38, 14,  6, 16, 4, 10, 10)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(50, 20,  8, 20, 4, 12, 12)
        self.b4 = Inception(64, 22,  9, 24, 4, 14, 16)

        self.a5 = Inception(76, 26,  12, 28, 4, 18, 18)
        self.b5 = Inception(90, 34,  16, 36, 6, 20, 20)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.dropout = nn.Dropout(p=0.3)
        self.linear = nn.Linear(6270, 2)

        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.pre_process_layer(x)
        x = self.first_layer(x)
        x = self.a3(x)
        x = self.b3(x)
        x = self.maxpool(x)
        x = self.a4(x)
        x = self.b4(x)
        x = self.maxpool(x)
        x = self.a5(x)
        x = self.b5(x)
        x = self.dropout(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return self.logsoftmax(x)


class RadarClassifier(LightningModule):
    def __init__(self):
        super().__init__()
        self.cnn = ConvNet()
        self.criterion = FocalLoss(alpha=0.5, gamma=2.0, reduction='mean')
        self.config = load_classifier_config()
        self.learning_rate = self.config['training']['learning_rates']
        self.batch_size = self.config['training']['batch_sizes']
        self.accuracy = Accuracy(num_classes=2)
        self.f1 = F1()
        self.recall = Recall()
        self.precision = Precision()



    def forward(self, x):
        return self.cnn(x)

    def prepare_data(self):
        dataset = MAFATDatasetAugmented([name for name in self.config['datasets'].keys() if name in ['train', 'train_spliced', 'synthetic']], self.config) 
        labels = dataset.target_type
        u, uc = np.unique(labels, return_index=False, return_inverse=False, return_counts=True, axis=None)
        n_val = int(len(dataset) * self.config['training']['fraction_of_examples_to_use_for_validation'])
        self.train_set, self.val_set = random_split(dataset, [len(dataset) - n_val, n_val])
        self.test_set = MAFATDataset(['test'], self.config)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=8, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=8, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=8, shuffle=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.cnn.parameters(), lr=self.learning_rate)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=0.0001, 
                                         threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
        return {'optimizer': optimizer,
                'lr_scheduler': lr_scheduler}

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        acc = self.accuracy(y_hat, y)
        f1 = self.f1(y_hat, y)
        recall = self.recall(y_hat, y)
        precision = self.precision(y_hat, y)

        pbar = {'ACC': acc, 'F1': f1, 'RECALL': recall, 'PRECISION': precision}
        return {'loss': loss, 'log': tensorboard_logs, 'progress_bar': pbar}

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['progress_bar']['ACC'] for x in outputs]).mean()
        avg_f1 = torch.stack([x['progress_bar']['F1'] for x in outputs]).mean()
        avg_recall = torch.stack([x['progress_bar']['RECALL'] for x in outputs]).mean()
        avg_precision = torch.stack([x['progress_bar']['PRECISION'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc, 'val_avg_f1': avg_f1, 'val_avg_recall': avg_recall, 'val_avg_precision': avg_precision}
        pbar = {'val_acc': avg_acc, 'val_avg_f1': avg_f1, 'val_avg_recall': avg_recall, 'val_avg_precision': avg_precision}
        return {'val_loss': avg_loss, 'log': tensorboard_logs, 'progress_bar': pbar}

    def test_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def deprecated_predict(self, test_loader, test):
        predictions = []
        self.cnn.eval()
        with torch.no_grad():
            for i, (iq_data, _) in enumerate(test_loader):  # forward + backward + optimize
                outputs = self.cnn(iq_data)
                predictions.extend(torch.exp(outputs.data[:, 1]).tolist())
            submission = pd.DataFrame()
            submission['segment_id'] = test.segment_ids
            submission['prediction'] = predictions
            submission['prediction'] = submission['prediction'].astype('float')
            # Save submission
            submission.to_csv(os.path.join(self.config['folders']['trained_models'],'SubmissionFiles', 'submission_{}.csv'.format(self.id)), index=False)

