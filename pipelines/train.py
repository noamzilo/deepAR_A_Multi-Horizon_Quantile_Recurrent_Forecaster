if __name__ == "__main__":
    print("hola")


import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


df = pd.read_csv("LD2011_2014.txt",
                 parse_dates=[0],
                 delimiter=";",
                 decimal=",")
df.rename({"Unnamed: 0": "timestamp"}, axis=1, inplace=True)


class ElDataset(Dataset):
    """Electricity dataset."""

    def __init__(self, df, samples):
        """
        Args:
            df: original electricity data (see HW intro for details).
            samples (int): number of sample to take per household.
        """
        df = df.resample('H', on='timestamp').mean().reset_index()
        self.raw_data = df.set_index('timestamp').asfreq('H')
        self._freq = self.raw_data.index.freq

        self._n_households = self.raw_data.shape[1]
        self._n_frames_per_house = self.raw_data.shape[0]

        self._max_allowed_timestamp_index = self._n_frames_per_house - 24 * (7 + 1)
        if self._max_allowed_timestamp_index < samples:
            print("warning, too many samples required!")
            samples = self._max_allowed_timestamp_index
        self._samples = samples
        self._time_stride = self._max_allowed_timestamp_index // self._samples

    def __len__(self):
        return self._samples * (self.raw_data.shape[1])

    def __getitem__(self, idx):
        if len(self) < idx:
            print(f"WARNING index {idx} out of bounds ({len(self)}")
        household, start_ts = self.get_mapping(idx)
        house_data = self.raw_data[household]
        hist_data = house_data.loc[start_ts:start_ts + (168 - 1) * self._freq]
        future_data = house_data.loc[start_ts + 168 * self._freq:start_ts + (168 + 24 - 1) * self._freq]
        torch_hist = torch.tensor(hist_data.values)
        torch_future = torch.tensor(future_data)
        return torch_hist, torch_future

    def get_mapping(self, idx):
        household_ind = idx % self._n_households
        time_sample_idx = idx // self._n_households
        start_ts_ind = time_sample_idx * self._time_stride

        household = self.raw_data.columns[household_ind]
        start_ts = self.raw_data.index[start_ts_ind]
        return household, start_ts
