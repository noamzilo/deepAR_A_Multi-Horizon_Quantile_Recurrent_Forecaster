import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch


class ElectricityLoadDataset(Dataset):
    """Sample data from electricity load dataset (per household, resampled to one hour)."""

    def __init__(self, df, samples, hist_num=168, fct_num=24):
        self.hist_num = hist_num
        self.fct_num = fct_num
        self.hist_len = pd.Timedelta(hours=hist_num)
        self.fct_len = pd.Timedelta(hours=fct_num)
        self.offset = pd.Timedelta(hours=1)
        self.samples = samples

        self.max_ts = df.index.max() - self.hist_len - self.fct_len + self.offset
        self.raw_data = df.copy()

        assert samples <= self.raw_data[:self.max_ts].shape[0]

        self.sample()

    def sample(self):
        """Sample individual series as needed."""

        # Calculate actual start for each household
        self.clean_start_ts = (self.raw_data != 0).idxmax()

        households = []

        for household in self.raw_data.columns:
            hh_start = self.clean_start_ts[household]
            hh_nsamples = min(self.samples, self.raw_data.loc[hh_start:self.max_ts].shape[0])

            hh_samples = (self.raw_data
                          .loc[hh_start:self.max_ts]
                          .index
                          .to_series()
                          .sample(hh_nsamples, replace=False)
                          .index)
            households.extend([(household, start_ts) for start_ts in hh_samples])

        self.samples = pd.DataFrame(households, columns=("household", "start_ts"))

        # Adding calendar features
        self.raw_data["yearly_cycle"] = np.sin(2 * np.pi * self.raw_data.index.dayofyear / 366)
        self.raw_data["weekly_cycle"] = np.sin(2 * np.pi * self.raw_data.index.dayofweek / 7)
        self.raw_data["daily_cycle"] = np.sin(2 * np.pi * self.raw_data.index.hour / 24)
        self.calendar_features = ["yearly_cycle", "weekly_cycle", "daily_cycle"]

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        household, start_ts = self.samples.iloc[idx]

        hs, he = start_ts, start_ts + self.hist_len - self.offset
        fs, fe = he + self.offset, he + self.fct_len

        hist_data = self.raw_data.loc[hs:, [household] + self.calendar_features].iloc[:self.hist_num]
        fct_data = self.raw_data.loc[fs:, [household] + self.calendar_features].iloc[:self.fct_num]

        return (torch.Tensor(hist_data.values),
                torch.Tensor(fct_data.values))