# MemoryLifeBench — Recommended GitHub Repository Structure

Repository layout for a paper on **learned memory lifecycle prediction** (importance, lifetime/TTL,
future utility, action) with an accompanying benchmark. The layout is designed so that every box in
the architecture figure maps to an identifiable module, and every number in the paper maps to a
script + config + seed.

---

## 1. Top-level tree

```
MemoryLifeBench/
│
├── README.md                     # what it is, install, 5-line quickstart, results table, citation
├── LICENSE                       # code license (MIT / Apache-2.0)
├── LICENSE-DATA                  # separate license for benchmark data (often CC BY-NC / CC BY-SA)
├── CITATION.cff                  # GitHub "Cite this repository" button
├── CHANGELOG.md
├── pyproject.toml                # package metadata + pinned deps (preferred over setup.py)
├── requirements.txt              # exact pinned versions (pip freeze of the env used for the paper)
├── environment.yml               # conda env incl. CUDA / faiss-gpu
├── Makefile                      # make setup / data / train / eval / paper
├── .gitignore
├── .gitattributes                # git-lfs rules for checkpoints, large jsonl
├── .pre-commit-config.yaml       # black, ruff, isort
├── .env.example                  # OPENAI_API_KEY / HF_TOKEN placeholders (never commit real keys)
│
├── .github/
│   └── workflows/
│       ├── ci.yml                # lint + unit tests + tiny smoke run on the sample data
│       └── reproduce-smoke.yml   # run_all.sh --smoke on 50 samples, catches broken pipelines
│
├── docs/
│   ├── architecture.md           # the figure, explained box by box
│   ├── benchmark_card.md         # datasheet: sources, sizes, splits, label definition, censoring
│   ├── annotation_guidelines.md  # if humans labelled reference events / importance
│   ├── reproducibility.md        # hardware, runtimes, seeds, expected variance
│   ├── FAQ.md
│   └── figures/
│       ├── architecture.pdf      # ← your uploaded figure
│       └── architecture.svg
│
├── data/                         # NO large raw files in git; only loaders + samples
│   ├── README.md                 # download links, checksums, licence of each source corpus
│   ├── download.sh               # pulls LoCoMo / LongMemEval / MSC / your dialogues
│   ├── checksums.sha256
│   ├── raw/                      # .gitignored
│   ├── interim/                  # .gitignored
│   ├── processed/                # .gitignored (train/val/test .jsonl with (T, δ) labels)
│   ├── splits/
│   │   ├── train_ids.txt
│   │   ├── val_ids.txt
│   │   └── test_ids.txt          # COMMIT these — makes splits reproducible without the data
│   └── samples/                  # 20–100 real committed examples so the repo runs out of the box
│       ├── dialogues_sample.jsonl
│       ├── memories_sample.jsonl
│       └── lifetime_events_sample.jsonl
│
├── configs/                      # Hydra / OmegaConf — one YAML per paper experiment
│   ├── config.yaml               # root defaults
│   ├── data/
│   │   ├── memorylifebench.yaml
│   │   ├── locomo.yaml
│   │   └── longmemeval.yaml
│   ├── encoder/
│   │   ├── e5_base.yaml
│   │   ├── bge_base.yaml
│   │   └── bge_large.yaml
│   ├── features/
│   │   ├── full.yaml
│   │   ├── no_contradiction.yaml   # ablation
│   │   ├── no_novelty.yaml         # ablation
│   │   └── no_temporal.yaml        # ablation
│   ├── model/
│   │   ├── ours_joint.yaml         # all 4 heads
│   │   ├── importance_only.yaml    # ablation
│   │   ├── survival_only.yaml      # ablation
│   │   └── fusion_variants.yaml    # concat / gated / cross-attention
│   ├── train/
│   │   ├── default.yaml            # lr, batch, epochs, loss weights, early stopping
│   │   └── sweep.yaml
│   ├── retrieval/
│   │   ├── sim_only.yaml           # baseline scoring
│   │   └── sim_importance_utility.yaml   # ours
│   ├── baselines/
│   │   ├── fifo.yaml
│   │   ├── lru.yaml
│   │   ├── heuristic_ttl.yaml
│   │   ├── generative_agents.yaml
│   │   ├── mem0.yaml
│   │   └── memgpt.yaml
│   └── eval/
│       ├── survival.yaml
│       ├── retrieval.yaml
│       └── downstream_qa.yaml
│
├── src/
│   └── memorylife/               # installable package: `pip install -e .`
│       ├── __init__.py
│       ├── registry.py           # name → class registry so configs stay declarative
│       │
│       ├── data/                            # ── data & benchmark construction ──
│       │   ├── build_benchmark.py           # dialogues → memory candidates → (T, δ) reference events
│       │   ├── event_labeling.py            # defines T = time-to-reference, δ = observed/censored
│       │   ├── censoring.py                 # right-censoring at session end / horizon
│       │   ├── preprocessing.py             # cleaning, segmentation, speaker/turn normalisation
│       │   ├── splits.py                    # user-disjoint splits, leakage checks
│       │   ├── datasets.py                  # torch Dataset / HF datasets wrappers
│       │   ├── collate.py
│       │   └── schema.py                    # pydantic schemas for every jsonl record
│       │
│       ├── encoders/                        # ── "Sentence Encoder" box ──
│       │   ├── base.py
│       │   ├── e5.py
│       │   ├── bge.py
│       │   └── cache.py                     # on-disk embedding cache (big runtime win)
│       │
│       ├── features/                        # ── "Semantic Feature Extractors" box ──
│       │   ├── base.py                      # common FeatureExtractor interface
│       │   ├── intent.py                    # fine-tuned intent classifier
│       │   ├── entities.py                  # GLiNER NER
│       │   ├── temporal.py                  # dates, deadlines, tense, expiry hints
│       │   ├── emotion.py                   # RoBERTa emotion / preference
│       │   ├── novelty.py                   # retrieval-conditioned: vs. retrieved memories
│       │   ├── contradiction.py             # NLI (DeBERTa-MNLI) vs. retrieved memories
│       │   └── pipeline.py                  # runs all extractors, returns feature dict
│       │
│       ├── fusion/                          # ── "Feature Fusion" box ──
│       │   ├── concat.py
│       │   ├── gated.py
│       │   ├── cross_attention.py
│       │   └── build.py                     # → fused vector z
│       │
│       ├── heads/                           # ── "Joint Lifecycle Predictor" box (4 heads) ──
│       │   ├── importance.py                # î ∈ [0,1]
│       │   ├── survival.py                  # hazard h(t|z) → S(t|z) → TTL
│       │   ├── future_utility.py            # P(retrieved in [t, t+Δ])
│       │   └── action.py                    # store / update / merge / forget
│       │
│       ├── models/
│       │   ├── joint_predictor.py           # encoder + features + fusion + 4 heads
│       │   ├── multitask.py                 # loss weighting / uncertainty weighting
│       │   └── checkpoint.py                # save/load, config fingerprint in the ckpt
│       │
│       ├── losses/                          # ── the methodological core; keep it separate ──
│       │   ├── censored_nll.py              # discrete-time survival NLL, censoring-aware
│       │   ├── cox_partial.py               # if you compare against Cox
│       │   ├── ranking.py                   # C-index surrogate / pairwise ranking
│       │   ├── importance_loss.py
│       │   ├── utility_loss.py
│       │   └── action_loss.py               # class-weighted CE
│       │
│       ├── memory/                          # ── "Memory Object" + "Memory Store" + lifecycle ──
│       │   ├── memory_object.py             # {text, embedding, î, TTL, type, action, provenance}
│       │   ├── store/
│       │   │   ├── base.py                  # abstract MemoryStore interface
│       │   │   ├── faiss_store.py
│       │   │   ├── chroma_store.py
│       │   │   └── sqlite_metadata.py
│       │   ├── reflection.py                # periodic importance decay, TTL expiry sweep
│       │   ├── compaction.py                # self-compaction: merge redundant → summary
│       │   ├── forgetting.py                # deletion policy driven by S(t|z) and î
│       │   └── audit.py                     # append-only log: every deletion logs its reason
│       │
│       ├── retrieval/                       # ── "Retriever" box (downstream use) ──
│       │   ├── retriever.py
│       │   ├── scoring.py                   # sim(q,e) + α·î + β·utility (weights in config)
│       │   ├── rerank.py
│       │   └── index.py
│       │
│       ├── inference/                       # ── "LLM → grounded answer" box ──
│       │   ├── pipeline.py                  # end-to-end: statement → memory → query → answer
│       │   ├── llm_client.py                # OpenAI / vLLM / HF, with response caching
│       │   └── prompts/
│       │       ├── qa_grounded.txt
│       │       ├── memory_extraction.txt
│       │       └── judge.txt                # LLM-as-judge rubric (commit the exact prompt!)
│       │
│       ├── evaluation/
│       │   ├── survival_metrics.py          # C-index, time-dependent AUC, Brier, IBS, calibration
│       │   ├── retrieval_metrics.py         # Recall@k, nDCG@k, MRR
│       │   ├── qa_metrics.py                # EM, F1, LLM-judge accuracy
│       │   ├── lifecycle_metrics.py         # forget precision/recall, wrongful-forget rate
│       │   ├── efficiency_metrics.py        # store size, tokens/query, latency, cost
│       │   ├── significance.py              # bootstrap CIs, paired t-test / Wilcoxon
│       │   └── report.py                    # metrics dict → LaTeX table
│       │
│       └── utils/
│           ├── seeding.py                   # seed_everything(), deterministic flags
│           ├── logging.py                   # wandb / tensorboard / plain jsonl logger
│           ├── io.py
│           ├── timer.py
│           └── viz.py                       # survival curves, calibration plots
│
├── scripts/                                 # thin CLI wrappers — no logic lives here
│   ├── download_data.sh
│   ├── preprocess.py
│   ├── build_benchmark.py                   # ← dataset generation code (reviewers ask for this)
│   ├── train.py
│   ├── evaluate.py
│   ├── run_baseline.py
│   ├── run_inference_demo.py                # single-user interactive demo
│   ├── export_tables.py                     # results/*.json → paper LaTeX tables
│   ├── export_figures.py                    # results → figures/*.pdf
│   ├── run_all.sh                           # FULL paper reproduction, ordered
│   └── run_smoke.sh                         # 3-minute version on data/samples/
│
├── baselines/                               # one file per compared system
│   ├── README.md                            # what was reimplemented vs. wrapped, and commit hashes
│   ├── no_forget.py                         # store everything (upper bound on storage)
│   ├── fifo.py
│   ├── lru.py
│   ├── heuristic_ttl.py                     # hand-tuned rules
│   ├── generative_agents_importance.py      # LLM-rated importance + recency decay
│   ├── mem0_wrapper.py
│   ├── memgpt_wrapper.py
│   ├── locomo.py
│   └── longmemeval.py
│
├── experiments/                             # provenance, not code
│   ├── README.md                            # experiment_id → paper table/figure mapping
│   ├── seeds.txt                            # e.g. 13, 42, 1337, 2024, 7
│   ├── main/                                # per-run: resolved config + metrics.json + logs
│   ├── ablation/
│   ├── sensitivity/                         # α, β, Δ, decay-rate sweeps
│   └── baseline_results/
│
├── results/
│   ├── tables/                              # table1_main.tex, table2_ablation.tex ...
│   ├── figures/                             # survival_curves.pdf, calibration.pdf ...
│   └── raw/                                 # metrics json per run, seed-level (for error bars)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_label_sanity_checks.ipynb         # censoring rate, T distribution
│   ├── 03_qualitative_examples.ipynb        # good/bad forgets for the paper appendix
│   └── 04_error_analysis.ipynb
│
└── tests/
    ├── test_schema.py
    ├── test_censoring.py                    # the loss must handle δ=0 correctly — test it
    ├── test_survival_loss.py                # closed-form checks on synthetic data
    ├── test_features.py
    ├── test_memory_store.py
    ├── test_audit_log.py                    # deletions always logged, never silent
    └── test_pipeline_smoke.py
```

---

## 2. Your checklist → where it lives

| Checklist item | Location |
|---|---|
| Proposed method | `src/memorylife/{features,fusion,heads,models,losses}` |
| Dataset preprocessing | `src/memorylife/data/preprocessing.py`, `scripts/preprocess.py` |
| Dataset generation code | `src/memorylife/data/build_benchmark.py`, `event_labeling.py`, `censoring.py` |
| Memory extraction | `src/memorylife/memory/memory_object.py`, `inference/prompts/memory_extraction.txt` |
| Retrieval code | `src/memorylife/retrieval/` |
| Evaluation scripts | `src/memorylife/evaluation/`, `scripts/evaluate.py` |
| Baseline implementations | `baselines/` |
| Training scripts | `scripts/train.py` + `configs/train/` |
| Inference script | `scripts/run_inference_demo.py`, `src/memorylife/inference/pipeline.py` |
| Config / hyperparameters | `configs/` (never hardcode in code) |
| Random seeds | `experiments/seeds.txt`, `utils/seeding.py`, seed field in every config |

**Additions beyond your list that reviewers reliably ask for:** the lifecycle/audit module (it is a
claimed contribution in the figure, so it must be code, not prose), `docs/benchmark_card.md`, the
exact LLM-judge prompt, split ID files, and per-seed raw metrics so error bars are verifiable.

---

## 3. Conventions worth enforcing

**Package, not scripts.** Put logic in `src/memorylife/`, install with `pip install -e .`. Scripts
should be ~30 lines: parse config, call library, dump metrics. This is what makes `baselines/` and
`tests/` able to import your method without path hacks.

**One config per paper row.** If Table 3 has 6 rows, there are 6 YAMLs. `scripts/export_tables.py`
should regenerate the LaTeX from `results/raw/` with zero manual editing.

**Every run writes a directory:** resolved config, git commit hash, seed, `metrics.json`, log file.
Name it `experiments/<group>/<expname>_seed<K>/`.

**Seeds:** fix 5 seeds, run everything with all 5, report mean ± std. Set them in
`utils/seeding.py` covering `random`, `numpy`, `torch`, `torch.cuda`, `PYTHONHASHSEED`, plus
`torch.use_deterministic_algorithms(True)` where feasible. Note remaining nondeterminism (FAISS, LLM
sampling — set `temperature=0` and cache responses) in `docs/reproducibility.md`.

**Data.** Do not commit raw corpora; commit `download.sh` + checksums + split ID files + a small
sample. If the sample is derived from a corpus with a restrictive licence, check redistribution terms
before committing it, and record the licence per source in `data/README.md`.

**Checkpoints.** Too big for git. Push to HuggingFace Hub or Zenodo (Zenodo gives a DOI, which is
good for the paper) and link from the README. Use git-lfs only if you must.

**Double-blind submission.** Keep the public repo anonymous during review — mirror to
`anonymous.4open.science`, strip author names from `LICENSE`, `CITATION.cff`, README, and check the
git history and notebook outputs for identifying paths.

---

## 4. Minimum viable README sections

1. One-paragraph description + the architecture figure
2. Install (conda + pip, CUDA version, ~2 commands)
3. Quickstart on `data/samples/` that runs in under 5 minutes
4. Data download instructions
5. Reproduce paper: `bash scripts/run_all.sh` + expected runtime & hardware
6. Results table with the numbers from the paper
7. Repo map (short version of the tree above)
8. Citation (BibTeX) + licence

---

## 5. Build order (so the repo works at every commit)

1. `data/schema.py` + `data/samples/` — fix the record format first, everything downstream depends on it
2. `data/build_benchmark.py` + `censoring.py` + `tests/test_censoring.py`
3. `encoders/` + `features/` + embedding cache
4. `losses/censored_nll.py` with unit tests on synthetic survival data
5. `models/joint_predictor.py` + `scripts/train.py`
6. `evaluation/survival_metrics.py` — verify the method works before wiring the memory system
7. `memory/` (store, reflection, compaction, forgetting, audit)
8. `retrieval/` + `inference/` + downstream QA eval
9. `baselines/`
10. `scripts/run_all.sh`, `export_tables.py`, README
