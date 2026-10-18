# An Agentic LLM System for Anomaly Detection and Automated Resolution in Cold-Chain Logistics

Code, data generators and results for the paper of the same name.

Four agents — a Monitor, a Diagnosis Agent, a Resolver and a Customer Agent —
are orchestrated as a LangGraph state graph. The system predicts temperature
excursions before they happen, identifies the cause, chooses a mitigation and
writes the stakeholder updates.

The more useful contribution is the evaluation. Detection lead time is measured
rather than assumed, false alarms are costed, and every configuration sees the
same simulated cargo under common random numbers. That protocol produced an
uncomfortable answer, which the paper reports: **the saving comes from automated
resolution, not from proactive detection.**

---

## Headline results

| | Result |
|---|---|
| Monitor, window level | precision 0.800, recall 0.802, F1 0.801, RMSE 1.525 °C |
| Monitor, event level | alarms on 100% of breaching events at threshold 0.80, 81.6% before the breach, median lead 10 min |
| Diagnosis | 79.17% with retrieval against 75.83% without, on 120 held-out cases — **not significant** (exact McNemar p = 0.541) |
| Resolver | 50/50 plans inside the 120 s target, median 57.9 s |
| Full system | 37.3% cheaper than reactive operation, 51.8% cheaper than a detection-only agentic system |
| Honest caveat | a plain threshold rule paired with the same automated pipeline is **equally cheap** |

A keyword classifier matches the 8B language model on this test set (79.17%
each, exact McNemar p = 1.000). Both findings are reported rather than buried;
Sect. 6.1 of the paper explains why the test set rewards lexical cues.

---

## Verifying the paper's numbers

Every quantitative claim in the paper is checked against the files that produced
it. No GPU and no model download are needed — the script only reads `outputs/`.

```bash
pip install pandas numpy scipy scikit-learn
python verify_paper_numbers.py
```

Expected output:

```
72 passed, 0 FAILED, 0 skipped, 0 source file(s) missing
Every paper number that can be checked against a file matches it.
```

The script then lists the handful of figures that their scripts print rather
than save, for manual confirmation.

---

## Repository layout

```
config.py                       all hyperparameters and paths
app.py                          demo entry point

monitor_agent/                  Informer model, predictor, runtime
diagnosis_agent/                ReAct agent, RAG retriever, test-set generator
resolver_agent/                 action simulator, runtime GA, resolver
customer_agent/                 message generation
orchestration/                  LangGraph workflow and orchestrator
core/                           LLM wrapper
baselines/                      B1-B5 baseline implementations
evaluation/                     simulator, metrics, experiment runner
data_generation/                synthetic generator, knowledge base builders

run_ablation_study.py           main experiment driver (Sect. 5.4, 5.5)
compare_baselines.py            baseline comparison (Sect. 5.4)
run_a2_experiment.py            GA weight optimisation (Sect. 5.6)
select_alarm_policy.py          alarm policy search (Sect. 4.2)
evaluate_diagnosis_accuracy.py  RAG vs no-RAG (Sect. 5.2)
evaluate_resolver.py            Resolver latency (Sect. 5.3)
val_selected_threshold_eval.py  Monitor window-level metrics (Sect. 5.1)
sensitivity_analysis.py         cost-weight sensitivity (Sect. 5.7)
verify_paper_numbers.py         checks 72 paper figures against outputs/
make_figures.py                 regenerates Figures 2 and 3 from outputs/

outputs/                        every result file behind the paper
figures/                        the figures as they appear in the paper
data/raw/                       three public source datasets, with provenance
data/knowledge_base/            retrieval store and diagnosis test sets
```

---

## Setup

```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # then add your own token
```

A Hugging Face token with access to `meta-llama/Llama-3.1-8B-Instruct` is needed
for the Diagnosis and Customer agents. Request access on the model page first.
**Never commit `.env`** — it is gitignored.

A GPU with 16 GB or more is needed for the two LLM agents. The Monitor, the
Resolver, the baselines and every ablation and sensitivity experiment run on CPU.

### Data

Three public source datasets are in `data/raw/`; see `data/raw/README.md` for
what each one was and was not used for. The generator does **not** read them —
its anomaly parameters come from Loisel et al. (2021) and WHO 2–8 °C storage
guidance, and are declared inline in `generate_cold_chain_training_data.py`. The
synthetic training arrays it writes are not tracked either — `cold_chain_X.npy`
alone is about 324 MB, over GitHub's per-file limit. Regenerate them with:

```bash
python generate_cold_chain_training_data.py
```

---

## Reproducing the results

Run these in order. The ordering matters and the warnings are not optional.

### 1. Monitor window-level metrics (Sect. 5.1)

```bash
python val_selected_threshold_eval.py
```

Prints precision 0.800, recall 0.802, F1 0.801, RMSE 1.525 °C. These four are
printed rather than saved, so `verify_paper_numbers.py` cannot check them.
Capture the console output.

### 2. Diagnosis accuracy (Sect. 5.2)

```bash
python evaluate_diagnosis_accuracy.py \
    --test_set data/knowledge_base/diagnosis_test_set_100.json \
    --tag _ts120
```

> **Both flags are required.** The script's default test set is the older
> 60-case file; the paper uses the 120-case one. The `_ts120` tag produces the
> CSVs the paper and the verification script actually read.

### 3. Alarm policy search (Sect. 4.2)

```bash
python select_alarm_policy.py --val_seeds 5
```

Searches 39 candidate policies over four triggers, nine thresholds and one to
three consecutive confirmations, on validation seeds 2000–2004. Selects
`classifier`, threshold 0.86, one confirmation. The runner-up at 0.88 costs 13.6
against 13.5; costs are compared after rounding to one decimal place.

### 4. Ablations, both thresholds (Sect. 5.4, 5.5)

```bash
python run_ablation_study.py --seeds 10 --n_anomalous 100 --n_normal 100 \
    --diag_test_set data/knowledge_base/diagnosis_test_set_100.json

python run_ablation_study.py --seeds 10 --n_anomalous 100 --n_normal 100 \
    --diag_test_set data/knowledge_base/diagnosis_test_set_100.json \
    --policy_file outputs/alarm_policy.json
```

Both runs print the Monitor's event-level figures, which are also console-only.

### 5. Baselines, both thresholds (Sect. 5.4)

```bash
python compare_baselines.py --seeds 10 --n_anomalous 100 --n_normal 100 \
    --diag_test_set data/knowledge_base/diagnosis_test_set_100.json

python compare_baselines.py --seeds 10 --n_anomalous 100 --n_normal 100 \
    --diag_test_set data/knowledge_base/diagnosis_test_set_100.json \
    --policy_file outputs/alarm_policy.json
```

### 6. Resolver weights (Sect. 5.6) — run this LAST

```bash
python run_a2_experiment.py --seeds 10 --val_seeds 5 \
    --n_anomalous 100 --n_normal 100 \
    --generations 30 --population 60 \
    --policy_file outputs/alarm_policy.json --evolve_per_cargo
```

> **Last, and in seconds mode.** `outputs/a2_summary.csv` carries no variant
> tag, so every A2 run overwrites it. If you run the
> `--action_time_unit minutes` robustness variant, re-run the command above
> afterwards to restore the seconds reading the paper reports.
> `--evolve_per_cargo` additionally writes `a2_per_cargo.csv`, which backs the
> re-evolved-per-valuation figures in Sect. 5.6.

### 7. Sensitivity and verification (Sect. 5.7)

```bash
python sensitivity_analysis.py
python verify_paper_numbers.py        # expect 72 passed, 0 FAILED
```

---

## Which file backs which number

| Paper section | Figures | Source |
|---|---|---|
| 4.2 | Alarm policy search, 39 candidates | `outputs/alarm_policy.json`, `alarm_policy_validation.csv` |
| 5.1 | Monitor window-level P/R/F1/RMSE | console output of `val_selected_threshold_eval.py` |
| 5.1 | Event-level detection and lead times | console output of `run_ablation_study.py`, both runs |
| 5.2 | Diagnosis accuracy, McNemar, per-class F1 | `outputs/diagnosis_results_WITH_RAG_ts120.csv`, `..._WITHOUT_RAG_ts120.csv` |
| 5.3 | Resolver latency | `outputs/resolver_evaluation_results.csv` |
| 5.4 | Baselines B1–B5 | `outputs/baselines_v3_summary.csv`, `baselines_v3_classifier_t0.86_k1_summary.csv` |
| 5.5 | Ablations A0–A4 | `outputs/ablation_v3_summary.csv`, `ablation_v3_classifier_t0.86_k1_summary.csv` |
| 5.6 | Evolved weights, price grid, per-cargo | `outputs/a2_summary.csv`, `a2_price_grid.csv`, `a2_weights.json`, `a2_per_cargo.csv` |
| 5.7 | Break-even analysis | `outputs/ablation_v3_sensitivity.csv`, `ablation_v3_classifier_t0.86_k1_cases.csv` |

Figures 2 and 3 read their values straight from these files:

```bash
python make_figures.py
```

---

## Reproducibility notes

**Seeding.** `run_a2_experiment.py` seeds Python's global `random` as well as the
NumPy generator, because DEAP's `cxBlend`, `mutGaussian` and `selTournament` all
draw from the global RNG. Without that the GA search is not reproducible.

**Decoding.** `Config.LLAMA_TEMPERATURE = 0.2`, not 0. Both arms of the retrieval
comparison use the same temperature so the comparison is fair, but repeated runs
vary slightly. Sect. 6.2 of the paper notes that this biases the paired test
towards the null, which makes the null result conservative.

**Common random numbers.** Every condition sees the same simulated cargo under
the same seed, so paired differences are not inflated by sampling noise. This is
why ten seeds give intervals as tight as they do.

**Diagnosis accuracy in the cost model.** `DIAG_ACCURACY` in
`run_ablation_study.py` is `{"rag": 0.7917, "no_rag": 0.7583}`, measured on the
120-case test set. An earlier version carried 0.7833 / 0.7000 from a superseded
60-case run; those values must not be reinstated.

**Which GA is which.** `resolver_agent/evolutionary_opt.py` is the runtime
optimiser used by the live pipeline, and its fitness is a simplified analytic
surrogate. The GA behind the Sect. 5.6 results lives inside
`run_a2_experiment.py` and scores weight vectors by simulated cost.

---

## Limitations

Stated here as they are in the paper, not buried.

- **The data is synthetic.** The generator is calibrated to WHO 2–8 °C storage
  guidance and to the anomaly parameters reported by Loisel et al. (2021) — not
  to measured operational data. Absolute costs should not be read as operational
  forecasts. What transfers is the comparison between conditions.
- **Retrieval did not help.** The hypothesis behind RQ2 is not supported on this
  test set. The retrieval store holds only 60 past cases.
- **A keyword classifier matches the language model** on diagnosis accuracy. The
  case descriptions were generated with the failure mode in view, so lexical
  cues are unusually strong.
- **Removing the Monitor makes the system cheaper** under the default
  false-alarm handling, by 6.3%. Sect. 5.7 quantifies exactly when that reverses.
- **Resolver latency rests on 50 scenarios.** 50 out of 50 within target tells
  you the rate is high, not how high.

---

## Live demo

```bash
python app.py --mode demo
streamlit run app.py
```

Runs the full four-agent pipeline on one simulated cargo and prints the Monitor
alarm, the diagnosis, the Resolver plan and the generated customer message.

---

## Citation

See `CITATION.cff`, or use the GitHub "Cite this repository" button.

## Copyright

Copyright (c) 2026 Mohammad Salahuddin. All rights reserved. No open-source licence is
granted — see `COPYRIGHT.md`. The repository is public so the results can be
inspected and reproduced; for any other use, please ask.

The three datasets in `data/raw/` remain under their original licences, which
are the uploaders' to set, not this repository's; see `data/raw/README.md` for
provenance and for the licence check that must happen before they are pushed.
