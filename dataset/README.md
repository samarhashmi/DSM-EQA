# DSM-EQA

**Physics-Informed Multimodal LLM for Intelligent Energy Question Answering**

Official dataset and code for:

> *"Physics-Informed Multimodal Large Language Models for Intelligent Energy
> Question Answering"*
> Samar · Waqas Amin — School of Information and Control Engineering, SWUST

---

## Overview

DSM-EQA is a benchmark and evaluation framework for energy-domain question
answering under hard physical constraints. Every sample in the dataset satisfies
three constraint classes taken directly from power-systems engineering:

| Constraint | Expression | Limit |
|---|---|---|
| Power balance | \|p\_gen − p\_load − p\_loss\| | ≤ 2% of p\_load |
| Voltage security | v\_min, v\_max | 0.95 – 1.05 pu |
| Thermal safety | u\_line | ≤ 1.00 pu |

The framework trains a multimodal LLM that jointly produces a fluent
natural-language answer **and** a physically feasible numerical output
vector, using these constraints as a differentiable regularisation term
during fine-tuning.

---

## Results

| Model | BLEU ↑ | ROUGE-L ↑ | BERTScore ↑ | PCR % ↑ | NA % ↑ | HR % ↓ | Expert ↑ |
|---|---|---|---|---|---|---|---|
| GPT-3.5-Turbo | 0.245 | 0.358 | 0.712 | 62.3 | 71.2 | 18.4 | 3.21 |
| GPT-4o (5-shot) | 0.301 | 0.423 | 0.771 | 69.4 | 78.6 | 11.3 | 3.74 |
| LLaMA-3-8B-Instruct | 0.289 | 0.401 | 0.759 | 71.8 | 77.4 | 14.7 | 3.61 |
| LLaMA-2-7B + DAPT | 0.312 | 0.421 | 0.768 | 73.8 | 79.1 | 12.4 | 3.82 |
| RAG Baseline | 0.278 | 0.389 | 0.741 | 66.1 | 74.2 | 16.8 | 3.55 |
| **DSM-EQA (ours)** | **0.387** | **0.512** | **0.841** | **94.6** | **89.4** | **3.2** | **4.32** |

*p < 0.001, Cohen's d = 0.82 (two-tailed paired t-test, Bonferroni corrected)*

PCR = Physics Compliance Rate · NA = Numerical Accuracy · HR = Hallucination Rate

---

## Repository layout

```
dsmeqa/
├── data/
│   ├── raw/
│   │   ├── qa_benchmark_full.csv          10 000 QA pairs (all categories)
│   │   ├── operational_timeseries_2020_2024.csv
│   │   ├── meteorological_weather_data.csv
│   │   ├── grid_topology_bus_data.csv
│   │   ├── grid_topology_branch_data.csv
│   │   ├── grid_topology_generator_data.csv
│   │   └── knowledge_base_metadata.csv
│   ├── processed/
│   │   ├── 01_demand_forecasting.csv      2 500 rows
│   │   ├── 02_supply_optimization.csv     2 000 rows
│   │   ├── 03_market_analysis.csv         1 500 rows
│   │   ├── 04_grid_operations.csv         1 500 rows
│   │   ├── 05_renewable_integration.csv   1 500 rows
│   │   ├── 06_anomaly_detection.csv       1 000 rows
│   │   └── master_metadata_index.csv
│   └── samples/
│       └── QA_samples.csv                 10 annotated examples
├── src/
│   ├── dataset.py                         PyTorch Dataset class
│   ├── physics_constraints.py             Differentiable penalty + checker
│   ├── evaluate.py                        Full metric suite
│   └── utils.py                           Loading and reporting helpers
├── tests/
│   └── test_physics.py                    Unit tests (10 tests, all pass)
├── requirements.txt
└── LICENSE
```

---

## Dataset

### Statistics

| Property | Value |
|---|---|
| Total QA pairs | 10 000 |
| Human-authored ratio | 23% (2 300 samples from operator query logs) |
| Template-generated, operator-validated | 77% (7 700 samples) |
| Physics compliance | 100% — all three constraints satisfied on every row |
| Bus systems | IEEE-14 (5%) · IEEE-30 (20%) · IEEE-118 (75%) |
| Renewable penetration range | 10% – 80% |
| Train / Val / Test split | 70 / 15 / 15 |
| Inter-rater agreement (construction) | Cohen's κ = 0.79 |
| Inter-rater agreement (hallucination annotation) | Cohen's κ = 0.81 |

### Task categories

| # | Category | Rows | Share |
|---|---|---|---|
| 1 | Demand Forecasting | 2 500 | 25% |
| 2 | Supply Optimization | 2 000 | 20% |
| 3 | Market Analysis | 1 500 | 15% |
| 4 | Grid Operations | 1 500 | 15% |
| 5 | Renewable Integration | 1 500 | 15% |
| 6 | Anomaly Detection | 1 000 | 10% |

### Question types

| Type | Share | Description |
|---|---|---|
| Factual | 40% | Direct numerical or operational fact retrieval |
| Reasoning | 40% | Single-chain causal or corrective reasoning |
| Multi-step | 20% | Multi-hop reasoning across dispatch, voltage, and thermal domains |

### Data generation

The QA benchmark was constructed through three sequential stages:

1. Physics-consistent numerical system states were generated under the three
   constraint classes stated in Section 4.4 of the paper, parameterised from
   published IEEE 14-, 30-, and 118-bus test case specifications.
   All 10 000 samples satisfy these constraints (verified programmatically).

2. Seventy-two question templates (12 per category) were instantiated with
   scenario-specific parameters spanning factual retrieval through multi-step
   reasoning.

3. Three certified grid operators (mean experience 11 ± 3 years) independently
   reviewed every candidate pair for physical feasibility and operational
   realism, discarding any pair failing either criterion (Cohen's κ = 0.79).
   The final benchmark comprises 23% directly human-authored questions drawn
   from anonymised operator query logs and 77% operator-validated template items.

### Column reference (QA benchmark files)

| Column | Description |
|---|---|
| `sample_id` | Unique identifier |
| `split` | train / val / test |
| `question_type` | factual / reasoning / multi_step |
| `question` | Natural-language query |
| `ground_truth_answer` | Expert-validated reference answer |
| `p_gen_MW` | Total system generation (MW) |
| `p_load_MW` | Total system load (MW) |
| `p_loss_MW` | Transmission losses (MW) |
| `v_min_pu` | Minimum bus voltage (pu) |
| `v_max_pu` | Maximum bus voltage (pu) |
| `u_line_pu` | Maximum line utilisation (pu) |
| `renewable_penetration_pct` | Instantaneous renewable share (%) |
| `bus_system` | IEEE-14 / IEEE-30 / IEEE-118 |
| `physics_valid` | True if all three constraints pass |
| `human_authored` | True if drawn from operator query logs |
| `expert_validated` | True if reviewed by certified grid operator |
| `bleu_score` | DSM-EQA BLEU on this sample |
| `bertscore` | DSM-EQA BERTScore on this sample |
| `hallucination_flag` | True if expert annotator flagged hallucination |

Anomaly detection file additionally includes `fault_type`, `fault_bus`,
`fault_severity`.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/dsm-eqa.git
cd dsm-eqa
pip install -r requirements.txt
```

---

## Quick start

```python
from src.dataset import DSMEQADataset
from src.physics_constraints import check_physics_compliance
from src.evaluate import DSMEQAEvaluator

# load the test partition
ds = DSMEQADataset("data/processed/", split="test")
ds.summary()

# inspect one sample
sample = ds[0]
print(sample["question"])
print(sample["ground_truth"][:200])
print(sample["physics_vector"])   # [p_gen, p_load, p_loss, v_min, v_max, u_line, ren]

# check a model prediction against the constraints
pred = dict(p_gen=4850.2, p_load=4823.4, p_loss=26.8,
            v_min=0.971, v_max=1.029, u_line=0.84)
compliant, violations = check_physics_compliance(pred)
print(compliant)     # True
print(violations)    # []

# run the full metric suite
evaluator = DSMEQAEvaluator()
results = evaluator.evaluate(
    predictions=model_answers,
    references=ground_truth_answers,
    numerical_preds=numerical_outputs,
    ground_truth_numerical=gt_numerical,
)
# BLEU: 0.387 | PCR: 94.6% | NA: 89.4% | HR: 3.2% | DSS: 91.9%
```

---

## Running the tests

```bash
python -m pytest tests/ -v
```

All 10 unit tests verify the physics constraint logic independently of the
full model stack.

---

## QA examples

### Demand forecasting — factual

**Operator:** What is the predicted peak demand for the Southwest region
on 14 July 2023?

**DSM-EQA:** The predicted peak demand for the Southwest region on
14 July 2023 is 4,823 MW, driven by elevated temperatures and high
cooling loads during the 13:00–17:00 afternoon window. Total generation
is scheduled at 4,850 MW to cover the ±5% forecast uncertainty band.
Voltage profiles across all 118 monitored buses remain within 0.95–1.05 pu
and no thermal violations are present.
`[p_gen=4850.2 MW | v_min=0.971 pu | u_line=0.84 pu ✓]`

---

### Grid operations — multi-step

**Operator:** N-1 contingency on line 47-69. What generation re-dispatch
restores security?

**DSM-EQA:** N-1 outage on line 47-69 creates a 340 MW loop flow
violation on the parallel path through buses 46-68-69, now loaded at
1.08 pu. Minimum-cost re-dispatch: reduce Unit 22 by 180 MW (580→400 MW)
and increase Unit 31 by 180 MW (240→420 MW). Lines 46-68 and 68-69
return to 0.88 and 0.91 pu respectively. Cost uplift is 4,140 USD/hour.
N-1 security confirmed on all 186 contingencies.
`[p_gen=4498.4 MW | v_min=0.966 pu | u_line=0.93 pu ✓]`

---

### Anomaly detection — critical multi-step

**Operator:** Cascading voltage collapse is imminent on the Northwestern
interface — voltages below 0.92 pu on four buses. What emergency actions
prevent a blackout?

**DSM-EQA:** Blackout prevention requires immediate sequential execution:
(1) shed 380 MW in load zones NW-2 and NW-3 within 60 seconds;
(2) insert 320 MVAR of reactive compensation within 90 seconds;
(3) reduce power imports from the South by 240 MW within 3 minutes;
(4) arm automatic voltage control relays on buses 87, 89, and 91.
Post-intervention: lowest bus voltage stabilises at 0.953 pu — above the
0.95 pu security limit.
`[p_gen=4218.4 MW | v_min=0.953 pu | u_line=0.97 pu ✓]`

---

## Expert validation

| Parameter | Value |
|---|---|
| Number of evaluators | 5 certified power system engineers |
| Mean experience | 14 ± 4 years |
| Evaluation protocol | Blinded to model identity |
| Inter-rater agreement | Krippendorff's α = 0.76 |
| Operator decision alignment | 91% on three real grid events |

---

## Citation

```bibtex
@article{samar2024dsmeqa,
  title   = {Physics-Informed Multimodal Large Language Models
             for Intelligent Energy Question Answering},
  author  = {Samar and Waqas Amin},
  journal = {Results in Engineering},
  year    = {2024},
  note    = {Under review}
}
```

---

## License

| Component | License |
|---|---|
| Dataset (CSV files) | CC BY 4.0 |
| Code (src/) | MIT |
| Operator query logs | Anonymised under utility data sharing agreement — raw logs not redistributable |

---

## Contact

Samar — `samar@swust.edu`
School of Information and Control Engineering, SWUST, Mianyang, China
