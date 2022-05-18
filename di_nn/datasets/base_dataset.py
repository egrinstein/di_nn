from bdb import Breakpoint
import pandas as pd
import torch
import torchaudio
import ast

from pathlib import Path

SR = 16000
N_MICROPHONE_SECONDS = 1
N_MICS = 4


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_dir,
                 sr=SR,
                 n_microphone_seconds=N_MICROPHONE_SECONDS,
                 n_mics=N_MICS,
                 metadata_dir=None):

        self.sr = sr
        self.n_microphone_seconds = n_microphone_seconds
        self.sample_duration = self.sr*self.n_microphone_seconds
        self.n_mics = n_mics
        self.metadata_dir = metadata_dir

        self.df = _load_dataframe(dataset_dir, metadata_dir)

    def __getitem__(self, index):
        if index >= len(self): raise IndexError
        
        sample_metadata = self.df.loc[index]

        # 1. Load signals
        x = torch.vstack([
            torchaudio.load(sample_metadata["signals_dir"] / f"{mic_idx}.wav")[0]
            for mic_idx in range(self.n_mics)
        ])
        #x = x[:, :self.sample_duration]

        # 2. Load low dimensional metadata

        y = sample_metadata.to_dict()
        y = _desserialize_lists_within_dict(y)

        # 3. Load metadata signals, if available
        if self.metadata_dir is not None:
            metadata = torch.vstack([
                torchaudio.load(sample_metadata["metadata_dir"] / f"{mic_idx}.wav")[0]
                for mic_idx in range(self.n_mics)
            ])
            y["metadata_signals"] =  metadata#[:, :self.sample_duration]

        return (x, y)

    def __len__(self):
        return self.df.shape[0]


def _load_dataframe(dataset_dir, metadata_dir=None):
    def _load(dataset_dir, metadata_dir):
        dataset_dir = Path(dataset_dir)
        df = pd.read_csv(dataset_dir / "metadata.csv")
        
        # 1. Get full paths to metadata signals dir, if available
        if metadata_dir is not None:
            metadata_dir = Path(metadata_dir)
            df["metadata_dir"] = df["signals_dir"].apply(
            lambda x: metadata_dir / x / x) # This is an ugly fix

        # 2. Get full paths to signals dir
        df["signals_dir"] = df["signals_dir"].apply(
            lambda x: dataset_dir / x)

        return df
    
    if type(dataset_dir) in [str, Path]:
        df = _load(dataset_dir, metadata_dir)
    else: # Multiple datasets
        dfs = [_load(d, m) for d, m in zip(dataset_dir, metadata_dir)]
        df = pd.concat(dfs, ignore_index=True)

    return df


def _desserialize_lists_within_dict(d):
    """Lists were saved in pandas as strings.
       This small utility function transforms them into lists again.
    """
    new_d = {}
    for key, value in d.items():
        if type(value) == str:
            try:
                new_value = ast.literal_eval(value)
                new_d[key] = torch.Tensor(new_value)
            except (SyntaxError, ValueError):
                new_d[key] = value
        else:
            new_d[key] = value
    return new_d
