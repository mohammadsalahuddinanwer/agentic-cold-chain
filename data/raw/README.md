# Source datasets


Three public datasets were examined during the project. **None of them
calibrates the synthetic generator, and none feeds any number reported in the
paper.** The generator's anomaly parameters derive from Loisel et al. (2021) and
WHO 2–8 °C storage guidance and are raised from there to increase the frequency
of failure events during training — the inline comments in
`generate_cold_chain_training_data.py` record which rates were scaled, and that
file reads no CSV. The reported
pipeline — `train_on_synthetic.py`, `compare_baselines.py`,
`run_ablation_study.py`, `sensitivity_analysis.py` and `run_a2_experiment.py` —
imports neither `data_generation/public_datasets.py` nor `data_loader/`, and
reads none of the three CSVs.

| File | Dataset | Rows | Licence | Where it came from |
|---|---|---|---|---|
| `euro_crop_agricultural_logistics.csv` | Euro Crop Agricultural Logistics | 53,305 | CC0 1.0 (public domain dedication) | https://www.kaggle.com/datasets/datasetengineer/euro-crop-agricultural-logistics-dataset |
| `smart_logistics_supply_chain.csv` | Smart Logistics Supply Chain | ~10,000 | CC0 1.0 (public domain dedication) | https://www.kaggle.com/datasets/ziya07/smart-logistics-supply-chain-dataset |
| `mendeley_cold_storage.csv` | A Multi-Parameter Dataset for Machine Learning Based Fruit Spoilage Prediction in an IoT-Enabled Cold Storage System | 10,995 | CC BY 4.0 (attribution required — see below) | https://data.mendeley.com/datasets/czz68d9fwj/1 |

## What each one was actually used for

- **Euro Crop Agricultural Logistics** — examined as a candidate training source
  and **rejected**. Its storage-temperature column has a median of 1,830, which
  is not a physical temperature. `check_units.py` reproduces that check, and the
  figure is the evidence for the sentence in Sect. 3.2 of the paper that
  explains why a public dataset could not be used. This is its only contribution
  to the paper. **Do not treat that column as temperature.**
- **Smart Logistics Supply Chain** and **Mendeley Cold Storage** — loaded only
  by `data_generation/public_datasets.py` and `data_loader/data_loader.py`, which
  belong to an earlier pipeline that the event-based generator described in
  Sect. 3.2 replaced. Neither dataset appears in any reported result, and
  neither is mentioned in the paper.

They are kept here so that the rejection in Sect. 3.2 can be re-checked rather
than taken on trust.

## Licences

**Both Kaggle datasets are CC0 1.0**, checked on their dataset pages. CC0 is a
public-domain dedication: the uploader has waived their copyright worldwide, and
the work may be copied, modified, distributed and performed, commercially
included, without permission. Redistributing those two CSVs here is therefore
permitted outright. CC0 asks for no attribution, but the provenance above is
given anyway as ordinary scholarly practice; nothing here implies that either
uploader endorses this work.

**The Mendeley dataset is CC BY 4.0.** Redistribution is permitted, but unlike
CC0 attribution is a condition of the licence: credit to the creators, the
title, a link to the licence, and an indication of whether changes were made.
The required notice is therefore given in full:

> **Source.** Nshekanabo Marius and Stanley Mugisha. *A Multi-Parameter Dataset
> for Machine Learning Based Fruit Spoilage Prediction in an IoT-Enabled Cold
> Storage System.* Mendeley Data, V1. doi: 10.17632/czz68d9fwj.1
>
> **Licence.** Creative Commons Attribution 4.0 International (CC BY 4.0),
> https://creativecommons.org/licenses/by/4.0/
>
> **Changes.** Redistributed here under the filename
> `mendeley_cold_storage.csv`. The data are otherwise as downloaded: 10,995
> rows and the six original columns, including the source file's own spellings
> `Light (Fux)` and `CO2 (pmm)`. No rows, columns or values were altered.
>
> Neither the creators nor Mendeley Data endorse this work or its use of the
> dataset.

The licence also notes that further permission may be required for any content
within a dataset identified as belonging to a third party. Nothing in this file
is so identified — it is six columns of sensor readings and a class label.
