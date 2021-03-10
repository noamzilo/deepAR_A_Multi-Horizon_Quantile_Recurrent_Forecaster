import matplotlib.pyplot as plt
import pandas as pd
import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from external_links.utills import DATA_DIR, download_dataset
from DeepArModel.ElectricityLoadModel import ElectricityLoadModel
from datasets.ElectricityLoadDataset import ElectricityLoadDataset
from data_modules.ElectricityLoadDataModule import ElectricityLoadDataModule

def set_pyplot_style():
    plt.style.use("bmh")
    plt.rcParams["figure.figsize"] = (6, 6)

# eldata = pd.read_parquet(DATA_DIR.joinpath("LD2011_2014.parquet"))

def main():

    data_path = os.path.join(DATA_DIR, "LD2011_2014.txt")

    if not os.path.isfile(data_path):
        download_dataset()

    assert os.path.isdir(DATA_DIR)
    assert os.path.isfile(data_path)

    eldata = pd.read_csv(data_path,
                         parse_dates=[0],
                         delimiter=";",
                         decimal=",")
    eldata.rename({"Unnamed: 0": "timestamp"}, axis=1, inplace=True)
    #
    # print(eldata.head())
    #
    eldata = eldata.resample("1H", on="timestamp").mean()
    #
    # (eldata != 0).mean().plot()
    # plt.ylabel("non-zero %")
    #
    eldata[eldata != 0].median().sort_values(ascending=False).plot(rot=90)
    # plt.yscale("log")
    #
    # plt.ylabel("magnitude")
    #
    # dataset = ElectricityLoadDataset(eldata, 100)
    #
    # hist, fct = dataset[4]
    #
    # print(f"hist.shape: {hist.shape}")
    # print(f"fct.shape: {fct.shape}")
    #
    # dataset.samples.groupby("household").size().unique()
    #
    # dm = ElectricityLoadDataModule(eldata)
    # dm.setup()
    #
    # assert dm.train_hh.intersection(dm.val_hh).empty
    # assert dm.train_hh.intersection(dm.test_hh).empty
    # assert dm.train_hh.size + dm.val_hh.size + dm.test_hh.size == 370
    #
    # x, y = next(iter(dm.train_dataloader()))
    #
    # print(f"x.shape: {x.shape}, y.shape: {y.shape}")
    #
    scaled_data = eldata / eldata[eldata != 0].mean() - 1

    dm = ElectricityLoadDataModule(scaled_data, batch_size=128)
    model = ElectricityLoadModel(lr=1e-3, hidden_units=64, num_layers=1)
    trainer = pl.Trainer(max_epochs=5, progress_bar_refresh_rate=1, gpus=1)
    trainer.fit(model, dm)

    # Example forecasts
    dm.setup(stage="test")

    batch = next(iter(dm.test_dataloader()))

    X, y = batch

    result = model.sample(X, 100)

    print(f"result.shape: {result.shape}")

    plt.plot(result.mean(dim=-1).numpy()[:, 8])
    plt.plot(y[8, :, 0])


if __name__ == "__main__":
    set_pyplot_style()
    main()
