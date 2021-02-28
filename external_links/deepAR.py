import matplotlib as plt


plt.style.use("bmh")
plt.rcParams["figure.figsize"] = (12, 4)

# %%

import pathlib

# %%

import numpy as np
import pandas as pd

# %%

DATA_DIR = pathlib.Path("data")

# %% md

# Loading data

# %%

df = pd.read_csv("data/AEP_hourly.csv.zip", parse_dates=["Datetime"], index_col="Datetime")


# Fix timestamps

df.index.is_monotonic, df.index.is_unique

# %%

df = df.sort_index()
df

# %% md

### New index

# %%

new_idx = pd.date_range("2004-10-01 01:00:00", "2018-08-03 00:00:00", freq="1H")

# %%

df[~df.index.duplicated(keep='first')]

# %%

dfi = df[~df.index.duplicated(keep='first')].reindex(new_idx)

# %%

dfi.index.is_monotonic, dfi.index.is_unique, dfi.index.freq

# %% md

### Missing values

# %%

dfi.isnull().mean()

# %%

dfi.ffill(inplace=True)

# %% md

# DalaModule

# %%

# PyTorch imports
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

# PyTorch Lightning imports
import pytorch_lightning as pl


# %%

class ElectricityDataset(Dataset):
    """Dataset which samples the data from hourly electricity data."""

    def __init__(self, df, samples, hist_len=168, fct_len=24, col="AEP_MW"):
        self.hist_num = hist_len
        self.fct_num = fct_len
        self.hist_len = pd.Timedelta(hours=hist_len)
        self.fct_len = pd.Timedelta(hours=fct_len)
        self.offset = pd.Timedelta(hours=1)

        self.max_ts = df.index.max() - self.hist_len - self.fct_len + self.offset
        self.raw_data = df

        assert samples <= self.raw_data[:self.max_ts].shape[0]
        self.samples = samples
        self.col = col
        self.sample()

    def sample(self):
        """Sample individual series as needed."""

        self.sample_idx = (self
                           .raw_data[:self.max_ts]
                           .index
                           .to_series()
                           .sample(self.samples, replace=False)
                           .index)

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        start_ts = self.sample_idx[idx]

        hs, he = start_ts, start_ts + self.hist_len - self.offset
        fs, fe = he + self.offset, he + self.fct_len

        hist_data = self.raw_data[hs:].iloc[:self.hist_num]
        fct_data = self.raw_data[fs:].iloc[:self.fct_num]

        return (torch.Tensor(hist_data[self.col].values),
                torch.Tensor(fct_data[self.col].values))


# %%

ds = ElectricityDataset(dfi, 10)

# %%

start_ts = ds.sample_idx[4]

# %%

dfi[start_ts:].head()

# %%

dfi[start_ts + pd.Timedelta(days=7):].head()

# %%

ds[4]


# %%

class ElectricityDataModule(pl.LightningDataModule):
    """DataModule for electricity data."""

    def __init__(self, df,
                 train_range=("2004", "2015"),
                 val_range=("2016", "2017"),
                 test_range=("2018", None),
                 factor=0.5,
                 batch_size=64,
                 workers=3):

        super().__init__()
        self.raw_data = df
        self.train_range = train_range
        self.val_range = val_range
        self.test_range = test_range
        self.factor = factor
        self.batch_size = batch_size
        self.workers = workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_df = self.raw_data[slice(*self.train_range)]
            val_df = self.raw_data[slice(*self.val_range)]

            self.train_ds = ElectricityDataset(train_df,
                                               samples=int(self.factor * train_df.shape[0]))
            self.val_ds = ElectricityDataset(val_df,
                                             samples=int(self.factor * val_df.shape[0]))

        if stage == "test" or stage is None:
            test_df = self.raw_data[slice(*self.test_range)]
            self.test_ds = ElectricityDataset(test_df,
                                              samples=int(self.factor * test_df.shape[0]))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.workers)


# %% md

# Encoder network

# %%

class ElectricityEncoder(pl.LightningModule):
    """Encoder network for encoder-decoder forecast model."""

    def __init__(self, hist_len=168, fct_len=24, num_layers=1, hidden_units=8):
        super().__init__()
        self.hist_len = hist_len
        self.fct_len = fct_len
        self.num_layers = num_layers
        self.hidden_units = hidden_units

        self.encoder = nn.LSTM(input_size=1,
                               hidden_size=self.hidden_units,
                               num_layers=self.num_layers,
                               batch_first=True)

    def forward(self, x):
        output, (henc, cenc) = self.encoder(x.view(x.shape[0], x.shape[1], 1))

        return output, henc, cenc


# %%

encoder = ElectricityEncoder()

# %%

hist_sample = torch.cat([ds[3][0].unsqueeze(0),
                         ds[5][0].unsqueeze(0)])

fct_sample = torch.cat([ds[3][1].unsqueeze(0),
                        ds[5][1].unsqueeze(0)])

output, hc, cc = encoder(hist_sample)

# %%

hist_sample.shape

# %%

hc.shape

# %%

output.shape

# %%

cc.shape


# %% md

# Decoder network

# %%

class ElectricityDecoder(pl.LightningModule):
    """Encoder network for encoder-decoder forecast model."""

    def __init__(self, hist_len=168, fct_len=24, num_layers=1, hidden_units=8):
        super().__init__()
        self.hist_len = hist_len
        self.fct_len = fct_len
        self.num_layers = num_layers
        self.hidden_units = hidden_units

        self.decoder = nn.LSTM(input_size=1,
                               hidden_size=self.hidden_units,
                               num_layers=self.num_layers,
                               batch_first=True)
        self.linear = nn.Linear(self.hidden_units, 1)

    def forward(self, x, hidden):
        output, (hc, cc) = self.decoder(x.view(x.shape[0], x.shape[1], 1), hidden)
        output = self.linear(output.squeeze(1))
        return output, hc, cc


# %%

decoder = ElectricityDecoder()

# %%

a, b, c = decoder(hist_sample[:, [-1]], (hc, cc))

# %%

decoder(a, (b, c))

# %%

a.shape


# %% md

# Encoder-decoder model

# %%

class ElectricityModel(pl.LightningModule):
    """Encoder network for encoder-decoder forecast model."""

    def __init__(self, hist_len=168, fct_len=24, num_layers=1, hidden_units=8, lr=1e-3):
        super().__init__()
        self.hist_len = hist_len
        self.fct_len = fct_len
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.lr = lr

        self.encoder = ElectricityEncoder(hist_len, fct_len, num_layers, hidden_units)
        self.decoder = ElectricityDecoder(hist_len, fct_len, num_layers, hidden_units)

    def forward(self, x):
        forecasts = []
        enc, hh, cc = self.encoder(x)
        enc = x[:, [-1]]

        for i in range(self.fct_len):
            enc, hc, cc = self.decoder(enc, (hh, cc))
            forecasts.append(enc)
        forecasts = torch.cat(forecasts, dim=1)
        return forecasts

    def training_step(self, batch, batch_idx):
        x, y = batch
        fct = self(x)
        return F.mse_loss(fct, y)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.mse_loss(logits, y)
        self.log('val_mse', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


# %% md

# Scaling

# %%

plt.figure(figsize=(6, 6))
dfi.plot(kind="hist", ax=plt.gca())

# %%

LIMH, LIML = 26e3, 9e3

# %%

plt.figure(figsize=(6, 6))
((2 * dfi - LIML - LIMH) / (LIMH - LIML)).plot(kind="hist", ax=plt.gca())

# %%

dfs = (2 * dfi - LIML - LIMH) / (LIMH - LIML)

ds = ElectricityDataModule(dfs, batch_size=32)
model = ElectricityModel(lr=1e-3, hidden_units=64)
trainer = pl.Trainer(max_epochs=20, progress_bar_refresh_rate=1, gpus=1)
trainer.fit(model, ds)

# %%

hist_sample_scaled = (2 * hist_sample - LIML - LIMH) / (LIMH - LIML)

# %%

plt.plot(((2 * hist_sample.numpy() - LIML - LIMH) / (LIMH - LIML))[0], label="historical data")
plt.plot(np.arange(168, 192, 1), model(hist_sample_scaled).detach().numpy()[0], label="forecast")
plt.plot(np.arange(168, 192, 1), ((2 * fct_sample.numpy() - LIML - LIMH) / (LIMH - LIML))[0], label="actual")

plt.legend(loc=0)
plt.tight_layout()

# %%

plt.plot(((2 * hist_sample.numpy() - LIML - LIMH) / (LIMH - LIML))[1], label="historical data")
plt.plot(np.arange(168, 192, 1), model(hist_sample_scaled).detach().numpy()[1], label="forecast")
plt.plot(np.arange(168, 192, 1), ((2 * fct_sample.numpy() - LIML - LIMH) / (LIMH - LIML))[1], label="actual")

plt.legend(loc=0)
plt.tight_layout()

# %%

dl = ds.test_dataloader()

# %%

xx = x[0]

# %%

xx.cuda()

# %%

for x in dl:
    fct = model(x[0].cuda())
    break

# %%

fct = fct.detach().cpu().numpy()

# %%

for stream in range(32):
    plt.figure(figsize=(12, 3))
    plt.plot((x[0][stream] * (LIMH - LIML) + (LIMH + LIML)) / 2, label="historical data")
    plt.plot(np.arange(168, 192, 1), (x[1][stream] * (LIMH - LIML) + (LIMH + LIML)) / 2, label="actual")
    plt.plot(np.arange(168, 192, 1), (fct[stream] * (LIMH - LIML) + (LIMH + LIML)) / 2, label="forecast")
    plt.legend(loc=0)
    plt.tight_layout()
    plt.show()

# %%


