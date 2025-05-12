import os
import glob
import torchaudio
import torch
from torch.utils.data import Dataset


class CustomMixDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000):
        self.root_dir = root_dir
        self.mix_dir = os.path.join(root_dir, "mix")
        self.s1_dir = os.path.join(root_dir, "s1")
        self.s2_dir = os.path.join(root_dir, "s2")
        self.sample_rate = sample_rate

        self.filenames = sorted([
            os.path.basename(f)
            for f in glob.glob(os.path.join(self.mix_dir, "*.wav"))
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        mix, _ = torchaudio.load(os.path.join(self.mix_dir, fname))
        s1, _ = torchaudio.load(os.path.join(self.s1_dir, fname))
        s2, _ = torchaudio.load(os.path.join(self.s2_dir, fname))

        # Stack sources along source dimension: [n_src, time]
        sources = torch.cat([s1, s2], dim=0)

        return mix, sources
