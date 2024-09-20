from pathlib import Path
import re

from einops import rearrange
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
import torchaudio as ta
from torchaudio.compliance.kaldi import fbank
from torchaudio.transforms import FrequencyMasking, TimeMasking

NON_LABELED_AUDIO_DIR = "~/Desktop/Audio_data/pretrain_Audio"
LABELED_AUDIO_DIR = "~/Desktop/Audio_data/finetune_Audio"
LABEL_DIR = "~/Desktop/Audio_data/finetune_Label_txt"
TARGET_PATH = "./data/Target_Label_20240606.csv"

AUDIO_LENGTH = 59.9         # s
WINDOW_SIZE = 3             # s
HOP_LENGTH = 0.5            # s
NUM_MEL_BIN = 128
MIN_FREQ = 100              # Hz
MAX_FREQ = 11000            # Hz
# NORM_MEAN = -7.0850544
# NORM_STD = 2.853875
NORM_MEAN = -6.679596900939941
NORM_STD = 2.211771011352539

class TWBird(IterableDataset):
    def __init__(self, src_file, labeled=False):
        super(TWBird, self).__init__()
        
        self.labeled = labeled

        self.file_paths = np.loadtxt(src_file, dtype=str)
        target_df = pd.read_csv(TARGET_PATH, sep=",", header=0)
        self.target = target_df["Label"].values.tolist()
        self.sub_target = [t.split("-")[0] + "-A" + t.split("-")[1] for t in self.target]

        self.window_num = np.floor((AUDIO_LENGTH - WINDOW_SIZE) // HOP_LENGTH)
        self.file_idx = 0
        self.window_idx = 0

    def __len__(self):
        return int(len(self.file_paths) * self.window_num)

    def __iter__(self):
        while True:
            if self.file_idx >= len(self.file_paths):
                self.file_idx = 0
            waveform, sr = ta.load(self.file_paths[self.file_idx], normalize=True)  # [2, T(s)*sr=points]
            waveform = waveform.mean(dim=0, keepdim=True)   # to mono [1, T(s)*sr=points]

            # slice audio into 3s windows with 0.5s hop length
            start_point = int(self.window_idx * HOP_LENGTH * sr)
            end_point = int(self.window_idx * HOP_LENGTH * sr + WINDOW_SIZE * sr)
            sliced_waveform = waveform[:, start_point : end_point]

            if self.labeled:
                # augmentation only when finetuning w/ labeled data
                sliced_waveform = self._roll_mag_aug(sliced_waveform)
                
            # get fbank features
            fbank_features = fbank(
                sliced_waveform, 
                htk_compat=True,
                sample_frequency=sr,
                window_type="hanning",
                num_mel_bins=NUM_MEL_BIN,
                low_freq=MIN_FREQ,
                high_freq=MAX_FREQ
            )

            if self.labeled:
                # augmentation only when finetuning w/ labeled data
                fbank_features = self._mask_time_freq_aug(fbank_features)
            
            # normalize fbank features
            fbank_features = (fbank_features - NORM_MEAN) / (NORM_STD * 2)

            if self.labeled:
                # add noise to fbank features
                fbank_features = self._add_noise_aug(fbank_features)

            fbank_features = rearrange(fbank_features, "f t -> 1 f t")
            fbank_features = F.pad(fbank_features, (0, 0, 22, 0))  # [1, 298, 128] -> [1, 320, 128]

            if self.labeled:
                soft_label = self._get_soft_label(
                    filename=Path(self.file_paths[self.file_idx]).stem,
                    start_time=(start_point / sr),
                    end_time=(end_point / sr)
                )
                if soft_label is None:
                    self.window_idx += 1
                    # if reach the end of the audio file, load the next file
                    if self.window_idx >= self.window_num:
                        self.file_idx += 1
                        self.window_idx = 0
                    continue
                yield fbank_features, torch.Tensor(soft_label)
            else:
                yield fbank_features, [-1] * (len(self.target)+1)

            self.window_idx += 1
            # if reach the end of the audio file, load the next file
            if self.window_idx >= self.window_num:
                self.file_idx += 1
                self.window_idx = 0

            if self.file_idx >= len(self.file_paths):
                self.file_idx = 0

    def _roll_mag_aug(self, waveform) -> torch.Tensor:
        waveform = waveform.numpy()
        # roll augmentation
        roll_idx = np.random.randint(0, waveform.shape[1])
        rolled_waveform = np.roll(waveform, roll_idx)
        # magnitude augmentation
        mag = np.random.beta(10, 10) + 0.5

        return torch.Tensor(rolled_waveform * mag)

    def _mask_time_freq_aug(self, spectrogram, mask_t=36, mask_f=36):
        # masking from frequency masking -> time masking
        f_mask = FrequencyMasking(mask_f)
        t_mask = TimeMasking(mask_t)

        spectrogram = rearrange(spectrogram, "t f -> 1 f t")
        masked_spectrogram = t_mask(f_mask(spectrogram))
        masked_spectrogram = rearrange(masked_spectrogram, "1 f t -> t f")

        return masked_spectrogram

    def _add_noise_aug(self, spectrogram):
        noisy_spectrogram = spectrogram + torch.rand(spectrogram.shape[0], spectrogram.shape[1]) * np.random.rand() / 10
        rolled_noisy_spectrogram = torch.roll(noisy_spectrogram, np.random.randint(-10, 10), 0)
        
        return rolled_noisy_spectrogram

    def _calculate_overlap(self, label_time:tuple, window_time:tuple) -> float:
        overlap_start, overlap_end = max(label_time[0], window_time[0]), min(label_time[1], window_time[1])
        overlap_duration = max(0, overlap_end - overlap_start)
        label_duration = label_time[1] - label_time[0]
        overlap_percentage = (overlap_duration / label_duration) if label_duration > 0 else 0
        return overlap_percentage
    
    def _get_soft_label(self, filename, start_time, end_time):
        labels_df = pd.read_csv(
            f"{LABEL_DIR}/{filename}.txt", sep="\t", header=None,
            names=["start_time", "end_time", "label"], dtype={"start_time": float, "end_time": float, "label": str}
        )
        labels_df["label"] = labels_df["label"].apply(lambda x: re.sub(r"\d", "", x))
        # select labels that are overlapped with the current window
        labels_df = labels_df[(labels_df["start_time"] <= end_time) & (labels_df["end_time"] >= start_time)]
        # if no label is found, return None
        if labels_df.empty:
            return None
        # if there are labels, calculate the soft label based on the overlap percentage and basic label
        soft_label = [1/(len(self.target)+1)] * (len(self.target)+1)
        for _, row in labels_df.iterrows():
            label = row["label"]
            overlap_percentage = self._calculate_overlap(label_time=(row["start_time"], row["end_time"]), window_time=(start_time, end_time))
            if label in self.target:
                soft_label[self.target.index(label)] += overlap_percentage
            elif label in self.sub_target:
                soft_label[self.sub_target.index(label)] += overlap_percentage * 0.6
            else:
                soft_label[-1] += overlap_percentage
        return soft_label

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = TWBird(src_file="./data/finetune/test.txt", labeled=True)
    # dataset = TWBird(src_file="./data/pretrain/train.txt")

    dataloader = DataLoader(dataset, batch_size=4, num_workers=4, pin_memory=True)
    for i, (feat, _) in enumerate(dataloader):
        print(feat.shape)
        print(len(dataloader))
        
        if i >= 3:
            break