# Source datasets


Three public datasets calibrate the synthetic generator. None is used directly
for training; the generator in `data_generation/` is fitted to their statistics
and to WHO 2–8 °C storage guidelines.

| File | Dataset | Rows | Where it came from |
|---|---|---|---|
| `euro_crop_agricultural_logistics.csv` | Euro Crop Agricultural Logistics | 53,305 | https://www.kaggle.com/datasets/datasetengineer/euro-crop-agricultural-logistics-dataset |
| `smart_logistics_supply_chain.csv` | Smart Logistics Supply Chain | ~10,000 | https://www.kaggle.com/datasets/ziya07/smart-logistics-supply-chain-dataset |
| `mendeley_cold_storage.csv` | Mendeley Cold Storage | 10,996 | https://data.mendeley.com/datasets/czz68d9fwj/1 |


## A note on the agricultural logistics dataset

It was examined early in the project and **rejected** as a training source: its
storage-temperature column has a median of 1,830, which is not a physical
temperature. It is retained only for the descriptive statistics reported in
Sect. 3.2 of the paper. Do not treat that column as temperature.
