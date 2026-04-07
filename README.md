# DSM-EQA: Physics-Informed Multimodal Large Language Models for Intelligent Energy Question Answering


> **Physics-Informed Multimodal LLM for Intelligent Energy Question Answering**


---

## Overview

DSM-EQA is a physics-informed multimodal large language model purpose-built for demand–supply management energy question answering. It seamlessly fuses operational time-series, meteorological data, and grid topology into a unified latent representation, adapts a pretrained language model via low-rank adaptation (LoRA), injects physics knowledge through **continuous prefix conditioning**, and simultaneously generates fluent natural-language explanations and quantitative system indicators under strict physical constraints enforced by differentiable regularization.

---

## Results

DSM-EQA achieves state-of-the-art performance across all evaluation metrics on the held-out test set (1,500 samples):

| Model | BLEU ↑ | ROUGE-L ↑ | BERTScore ↑ | PCR (%) ↑ | NA (%) ↑ | HR (%) ↓ | DSA (%) ↑ |
|---|---|---|---|---|---|---|---|
| GPT-3.5-Turbo | 0.245 | 0.358 | 0.712 | 62.3 | 71.2 | 18.4 | 58.1 |
| GPT-4o (5-shot) | 0.301 | 0.427 | 0.781 | 74.2 | 78.4 | 11.3 | 70.6 |
| LLaMA-2-7B-Base | 0.156 | 0.241 | 0.634 | 48.7 | 58.3 | 31.2 | 44.2 |
| LLaMA-2-7B-Chat | 0.198 | 0.312 | 0.689 | 55.1 | 64.8 | 24.6 | 51.3 |
| LLaMA-3-8B-Instruct | 0.271 | 0.398 | 0.749 | 67.8 | 74.1 | 14.6 | 63.4 |
| LLaMA-2-7B + FT | 0.267 | 0.394 | 0.745 | 68.4 | 76.5 | 15.2 | 64.7 |
| LLaMA-2-7B + DAPT | 0.312 | 0.421 | 0.768 | 73.8 | 79.1 | 12.4 | 70.9 |
| Energy-BERT | 0.189 | 0.298 | 0.701 | 71.2 | 68.9 | 21.3 | 60.2 |
| RAG Baseline | 0.289 | 0.411 | 0.752 | 69.3 | 75.8 | 13.7 | 66.1 |
| **DSM-EQA (Proposed)** | **0.387** | **0.512** | **0.841** | **94.6** | **89.4** | **3.2** | **87.3** |

| Comparison | BLEU | ROUGE-L | BERTScore | PCR | NA | HR | DSA |
|---|---|---|---|---|---|---|---|
| vs. Best Baseline | +24.1% | +21.6% | +9.5% | +28.2% | +13.0% | −74.8% | +23.1% |
| vs. GPT-3.5 | +58.0% | +43.0% | +18.1% | +51.8% | +25.6% | −82.6% | +50.3% |
| vs. GPT-4o | +28.6% | +19.9% | +7.7% | +27.5% | +14.0% | −71.7% | +23.7% |

*p < 0.001, Cohen's d = 0.82 (two-tailed paired t-test, Bonferroni corrected)*

**PCR** = Physics Compliance Rate · **NA** = Numerical Accuracy · **HR** = Hallucination Rate · **DSA** = Decision Support Accuracy

### Task-Wise Performance

| Task Category | DSA (%) | Notes |
|---|---|---|
| Demand Forecasting | 91.2 | Best performance — effective encoding of temporal dependencies and load patterns |
| Supply Optimization | 88.7 | Strong — benefits from multimodal physics grounding |
| Market Analysis | 86.4 | Competitive — leverages domain-adaptive pretraining |
| Grid Operations | 87.9 | Reliable — topology encoding supports dispatch reasoning |
| Renewable Integration | 85.3 | Robust — weather modality integration is key |
| Anomaly Detection | 83.1 | Lowest — challenging fine-grained fault discrimination shared by all models |

---

## Ablation Study

Systematic validation of each design component on the held-out test set:

| Configuration | BLEU | BERTScore | PCR (%) | NA (%) |
|---|---|---|---|---|
| w/o Multimodal | 0.324 | 0.782 | 88.2 | 81.7 |
| w/o Physics Loss | 0.371 | 0.829 | 71.3 | 85.2 |
| w/o DAPT | 0.298 | 0.754 | 89.1 | 82.4 |
| w/o Time Series | 0.351 | 0.807 | 90.8 | 83.6 |
| Lower LoRA Rank | 0.362 | 0.823 | 92.1 | 87.3 |
| **Full Model** | **0.387** | **0.841** | **94.6** | **89.4** |

---

## Computational Efficiency

| Metric | Value |
|---|---|
| Average response latency | 1.8 seconds per query |
| Total training time | 84 hours (single NVIDIA A100 80GB) |
| 4-bit quantized inference | ~97% of full-precision performance retained |
| Deployment | Runnable on consumer-grade hardware with quantization + LoRA |

---

## Error Analysis

Hallucination rate (3.2%) on 500 stratified test responses decomposed as follows:

| Error Category | Share | Description |
|---|---|---|
| Minor numerical errors | 42% | Small deviations within tolerance, no safety impact |
| Incomplete contextual information | 31% | Missing or insufficient context for full accuracy |
| Outdated data reference | 18% | Reliance on historical patterns not fully reflecting current conditions |
| Model-level failures | 9% | Genuine reasoning or generation errors |

> **Safety note:** None of the hallucination instances result in safety violations. All generated recommendations remain in full compliance with grid reliability requirements.

---



## Comparison with Related Work

| Method | Multimodal Fusion | Physics Constraints | Differentiable Training | NLQA | Energy Domain | Real-world Val. |
|---|---|---|---|---|---|---|
| AFDFE-LLM | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ |
| PowerGridQA | ✗ | ✗ | ✗ | ✓ | ✓ | ~ |
| GAIA | ✗ | ✗ | ✗ | ✓ | ✓ | ~ |
| PowerGraph-LLM | ~ | ~ | ✗ | ✗ | ✓ | ~ |
| PINCO | ✗ | ✓ | ✓ | ✗ | ✓ | ✗ |
| LLM-Secure | ✗ | ~ | ✗ | ~ | ✓ | ~ |
| GPT-3.5 | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| LLaMA-2 | ✗ | ✗ | ✗ | ✓ | ~ | ✗ |
| **DSM-EQA (Proposed)** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

---

## Dataset

### Statistics

| Property | Value |
|---|---|
| Total QA pairs | 10,000+ |
| Human-authored ratio | 23% (2,300 samples from anonymized operator query logs) |
| Template-generated, operator-validated | 77% (7,700 samples) |
| Physics compliance | 100% — all three constraints satisfied on every row |
| Bus systems | IEEE-14 (5%) · IEEE-30 (20%) · IEEE-118 (75%) |
| Renewable penetration range | 10% – 80% |
| Train / Val / Test split | 70 / 15 / 15 (test set = 1,500 samples) |
| Inter-rater agreement (construction) | Cohen's κ = 0.79 |
| Inter-rater agreement (hallucination annotation) | Cohen's κ = 0.81 |
| Template accuracy (held-out test) | 85.6% on template items vs. 84.7% on human-authored (0.9% gap) |

### Task Categories

| # | Category | Proportion (%) | Approx. Samples |
|---|---|---|---|
| 1 | Demand Forecasting | 25 | 2,500 |
| 2 | Supply Optimization | 20 | 2,000 |
| 3 | Market Analysis | 15 | 1,500 |
| 4 | Grid Operations | 15 | 1,500 |
| 5 | Renewable Integration | 15 | 1,500 |
| 6 | Anomaly Detection | 10 | 1,000 |

### Question Types

| Type | Share | Description |
|---|---|---|
| Factual | 40% | Direct numerical or operational fact retrieval |
| Reasoning (single-chain) | 40% | Causal or corrective reasoning |
| Multi-step (multi-hop) | 20% | Multi-hop reasoning across dispatch, voltage, and thermal domains |

### Data Generation

The QA benchmark was constructed through sequential stages:

1. **Physics-consistent system states:** 10,000 states were generated by sampling load variability (±30%), renewable penetration (10–80%), and fault scenarios, all verified against power balance, voltage (0.95–1.05 pu), and thermal limits (≤ 1.00 pu).

2. **Template instantiation:** 72 structured templates (12 per category) were instantiated at three cognitive levels — factual (40%), single-chain reasoning (40%), and multi-hop reasoning (20%).
---

## Repository Layout

```
dsmeqa/
├── data/
│   ├── raw/
│   │   ├── qa_benchmark_full.csv          10,000+ QA pairs (all categories)
│   │   ├── operational_timeseries_2020_2024.csv
│   │   ├── meteorological_weather_data.csv
│   │   ├── grid_topology_bus_data.csv
│   │   ├── grid_topology_branch_data.csv
│   │   ├── grid_topology_generator_data.csv
│   │   └── knowledge_base_metadata.csv
│   ├── processed/
│   │   ├── 01_demand_forecasting.csv      2,500 rows
│   │   ├── 02_supply_optimization.csv     2,000 rows
│   │   ├── 03_market_analysis.csv         1,500 rows
│   │   ├── 04_grid_operations.csv         1,500 rows
│   │   ├── 05_renewable_integration.csv   1,500 rows
│   │   ├── 06_anomaly_detection.csv       1,000 rows
│   │   └── master_metadata_index.csv
│   └── samples/
│       └── QA_samples.csv                 10 annotated examples
├── config/            Configuration files
├── dataset/           Dataset handling and preprocessing
├── dsm-eqa/           Core project module
├── evaluation/        Evaluation scripts and metrics
├── models/            Model architectures
├── training/          Training pipelines
├── utils/             Utility/helper functions
├── main.py            Entry point of the project
├── README.md          Project documentation
├── requirements.txt   Dependencies
└── setup.py           Package setup file
```

---

## Installation

### Prerequisites

- Python 3.10
- PyTorch 2.1.0
- NVIDIA GPU (recommended: A100 80GB; consumer-grade GPU supported with 4-bit quantization)

```bash
git clone https://github.com/samarhashmi/DSM-EQA.git
cd DSM-EQA
pip install -r requirements.txt
```

### Dependencies

| Library | Version | Purpose |
|---|---|---|
| PyTorch | 2.1.0 | Deep learning backend |
| transformers | 4.36.0 | Pretrained model integration |
| PEFT | 0.7.1 | Parameter-efficient fine-tuning (LoRA) |

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Backbone LLM | LLaMA-2-7B |
| LoRA rank (r) | 16 |
| LoRA α | 32 |
| LoRA dropout | 0.05 |
| Learning rate | 2 × 10⁻⁴ (cosine annealing) |
| Optimizer | AdamW (β₁ = 0.9, β₂ = 0.999) |
| Batch size | 16 |
| Max sequence length | 512 tokens |
| Training epochs | 10 |
| Physical loss weight (λ_phys) | 0.3 |
| Numerical loss weight (λ_num) | 0.2 |
| Prefix length (P) | 20 tokens |
| GNN layers | 3 |
| Hidden dimension (d) | 768 |
| Attention heads | 8 (d_k = 64) |
| GPU | NVIDIA A100 80GB |
| Training time | 84 hours |

---

## QA Examples

### Demand Forecasting — Factual

**Operator:** What is the predicted peak demand for the Southwest region on 14 July 2023?

**DSM-EQA:** The predicted peak demand for the Southwest region on 14 July 2023 is 4,823 MW, driven by elevated temperatures and high cooling loads during the 13:00–17:00 afternoon window. Total generation is scheduled at 4,850 MW to cover the ±5% forecast uncertainty band. Voltage profiles across all 118 monitored buses remain within 0.95-1.05 pu and no thermal violations are present.

`[p_gen=4850.2 MW | v_min=0.971 pu | u_line=0.84 pu ✓]`

---

### Grid Operations — Multi-Step

**Operator:** N-1 contingency on line 47-69. What generation re-dispatch restores security?

**DSM-EQA:** N-1 outage on line 47-69 creates a 340 MW loop flow violation on the parallel path through buses 46-68-69, now loaded at 1.08 pu. Minimum-cost re-dispatch: reduce Unit 22 by 180 MW (580→400 MW) and increase Unit 31 by 180 MW (240→420 MW). Lines 46-68 and 68-69 return to 0.88 and 0.91 pu respectively. Cost uplift is 4,140 USD/hour. N-1 security confirmed on all 186 contingencies.

`[p_gen=4498.4 MW | v_min=0.966 pu | u_line=0.93 pu ✓]`

---

### Anomaly Detection — Critical Multi-Step

**Operator:** Cascading voltage collapse is imminent on the Northwestern interface voltages below 0.92 pu on four buses. What emergency actions prevent a blackout?

**DSM-EQA:** Blackout prevention requires immediate sequential execution:

1. Shed 380 MW in load zones NW-2 and NW-3 within 60 seconds
2. Insert 320 MVAR of reactive compensation within 90 seconds
3. Reduce power imports from the South by 240 MW within 3 minutes
4. Arm automatic voltage control relays on buses 87, 89, and 91

Post-intervention:
- Lowest bus voltage stabilises at 0.953 pu, above the 0.95 pu security limit




## Evaluation Metrics

| Metric | Definition |
|---|---|
| **DSA** (Decision Support Accuracy) | Binary correctness: response correct only if numerical output within ±5% of ground truth AND recommended action matches operator reference |
| **PCR** (Physics Compliance Rate) | Percentage of responses where all three physical constraints (power balance, voltage, thermal) are satisfied |
| **NA** (Numerical Accuracy) | Mean relative error between predicted and ground-truth numerical indicators |
| **HR** (Hallucination Rate) | Percentage of responses flagged for physically impossible values, factually incorrect claims, or fabricated grid entities |
| **BLEU** | Bilingual Evaluation Understudy — measures lexical fidelity |
| **ROUGE-L** | Recall-Oriented Understudy — longest common subsequence for overlap |
| **BERTScore** | Contextual embedding similarity for semantic alignment |

---

## Limitations

- **Synthetic data dependency:** The model is trained on synthetic data, which may limit direct deployment to operational environments without fine-tuning on real utility data.
- **Historical condition reliance:** Performance under future high-renewable, high-electrification regimes remains unverified as grid behaviors may shift significantly with growing renewable penetration and more frequent extreme weather events.
- **Generalization gap:** Template-generated QA pairs (77%) introduce a modest generalization gap, reflected in the accuracy difference between template items (85.6%) and human-authored queries (84.7%).
- **Regional applicability:** Applicability across regions with different market structures, regulatory policies, and trading mechanisms requires further validation.

---

## Future Research Directions

- **Causal inference** to differentiate causal links from statistical correlations
- **Multi-agent coordination** in federated learning environments
- **Language-procedural reasoning** combined with reinforcement learning for autonomous grid control
- **Real-time adaptive learning** mechanisms for evolving grid conditions
- **Cross-domain transfer learning** to other safety-critical infrastructure
- **Explainability and uncertainty awareness** for enhanced operator confidence

---

## Citation

```bibtex
@article{abbas2024dsmeqa,
  title   = {Physics-Informed Multimodal Large Language Models for Intelligent Energy Question Answering},
  author  = {Samar Abbas and Waqas Amin},
  journal = {Results in Engineering},
  year    = {2026},
  note    = {under-review}
}
```


