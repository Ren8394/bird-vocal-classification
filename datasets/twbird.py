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
NUM_MEL_BIN = 128
MIN_FREQ = 100              # Hz
MAX_FREQ = 11000            # Hz
# NORM_MEAN = -7.0850544
# NORM_STD = 2.853875
NORM_MEAN = -6.679596900939941
NORM_STD = 2.211771011352539

class TWBird(IterableDataset):
    def __init__(self, src_file, labeled=False,
                 window_size=3.0, hop_length=0.5
                 ):
        super(TWBird, self).__init__()
        
        self.labeled = labeled
        self.window_size = window_size
        self.hop_length = hop_length

        self.file_paths = np.loadtxt(src_file, dtype=str)
        target_df = pd.read_csv(TARGET_PATH, sep=",", header=0)
        self.target = target_df["Label"].values.tolist()
        self.sub_target = [t.split("-")[0] + "-A" + t.split("-")[1] for t in self.target]

        self.window_num = int(np.floor((AUDIO_LENGTH - window_size) / hop_length))
        self.total_samples = len(self.file_paths) * self.window_num

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:  # single-process data loading
            start_idx = 0
            end_idx = len(self.file_paths)
        else:  # in a worker process
            per_worker = int(np.ceil(len(self.file_paths) / worker_info.num_workers))
            worker_id = worker_info.id
            start_idx = worker_id * per_worker
            end_idx = min(start_idx + per_worker, len(self.file_paths))

        for file_idx in range(start_idx, end_idx):
            waveform, sr = ta.load(self.file_paths[file_idx], normalize=True)
            waveform = waveform.mean(dim=0, keepdim=True)

            for window_idx in range(self.window_num):
                start_point = int(window_idx * self.hop_length * sr)
                end_point = int(window_idx * self.hop_length * sr + self.window_size * sr)
                sliced_waveform = waveform[:, start_point : end_point]

                if self.labeled:
                    sliced_waveform = self._roll_mag_aug(sliced_waveform)
                
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
                    fbank_features = self._mask_time_freq_aug(fbank_features)
                
                fbank_features = (fbank_features - NORM_MEAN) / (NORM_STD * 2)

                if self.labeled:
                    fbank_features = self._add_noise_aug(fbank_features)

                fbank_features = rearrange(fbank_features, "f t -> 1 f t")
                if self.window_size == 3.0 and self.hop_length == 0.5:
                    fbank_features = F.pad(fbank_features, (0, 0, 11, 11))
                elif self.window_size == 1.0 and self.hop_length == 0.5:
                    fbank_features = F.pad(fbank_features, (0, 0, 15, 15))

                if self.labeled:
                    soft_label = self._get_soft_label(
                        filename=Path(self.file_paths[file_idx]).stem,
                        start_time=(start_point / sr),
                        end_time=(end_point / sr)
                    )
                    if soft_label is None:
                        continue
                    yield fbank_features, torch.Tensor(soft_label)
                else:
                    yield fbank_features, torch.Tensor([-1] * len(self.target))

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
        # Select files accepted NOTA label (skip)
        with_nota = True

        # read labels from the txt file
        labels_df = pd.read_csv(
            f"{LABEL_DIR}/{filename}.txt", sep="\t", header=None,
            names=["start_time", "end_time", "label"], dtype={"start_time": float, "end_time": float, "label": str}
        )
        labels_df["label"] = labels_df["label"].apply(lambda x: re.sub(r"\d", "", x))   # remove numbers in the label

        # select labels that are overlapped with the current window
        labels_df = labels_df[(labels_df["start_time"] <= end_time) & (labels_df["end_time"] >= start_time)]
        if with_nota:
            soft_label = [1/(len(self.target)+1)] * (len(self.target)+1)  # [target, NOTA]  
        else:
            soft_label = [1/(len(self.target))] * (len(self.target))  # [target]

        # if no label is found, set the last element NOTA to 1
        if labels_df.empty:
            if with_nota:
                soft_label[-1] = 1
        # if there are labels, calculate the soft label based on the overlap percentage and basic label
        else:
            for _, row in labels_df.iterrows():
                label = row["label"]
                overlap_percentage = self._calculate_overlap(label_time=(row["start_time"], row["end_time"]), window_time=(start_time, end_time))
                # label in the target list
                if label in self.target:
                    soft_label[self.target.index(label)] += overlap_percentage
                # label in the sub_target list
                elif label in self.sub_target:
                    soft_label[self.sub_target.index(label)] += overlap_percentage * 0.6
                # NOTA label
                else:
                    if with_nota:
                        soft_label[-1] += overlap_percentage

        # normalize the soft label, make every element in the list to be in the range of [0, 1]
        # min-max normalization
        min_val = min(soft_label)
        max_val = max(soft_label)
        soft_label = [(l - min_val) / (max_val - min_val) for l in soft_label]

        # clip the value to be in the range of [0, 1]
        # soft_label = [max(0, min(1, l)) for l in soft_label]

        return soft_label

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # dataset = TWBird(src_file="./data/finetune/test.txt", labeled=True)
    dataset = TWBird(src_file="./data/pretrain/train.txt", window_size=1.0, hop_length=0.5)

    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True)
    for i, (feat, l) in enumerate(dataloader):
        print(feat.shape)
        # print(l.shape)
        # print(len(dataloader))
        # print(l)

        if i >= 1:
            break