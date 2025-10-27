# Configuration-Driven Experiments

The `configs/` tree captures reusable settings for datasets and experiment runs so we can repeat larger evaluations without memorising long CLI invocations.

- `configs/datasets/` — YAML files that describe how to download, unpack, and address a corpus for a given adapter. Each config maps cleanly onto the `DatasetConfig` model (`src/models/configs.py`). When a dataset config is referenced, `scripts/run_evals.py --config` will download assets (if needed) before ingestion.
- `configs/experiments/` — Experiment presets that bundle retriever choices, artifact locations, evaluation assets, and (optionally) dataset configs. These hydrate the CLI and still allow per-run overrides on the command line.

## Using an Experiment Config

```bash
uv run python scripts/run_evals.py --config configs/experiments/hotpotqa_llm_vs_vector.yaml
```

The loader applies values from the experiment file first, then merges in any CLI overrides you provide. Matching CLI flags always win if you pass them explicitly.

Key fields you can set inside an experiment config:

- `dataset_configs`: list of dataset config paths (relative to the experiment file) to materialise before ingestion.
- Top-level storage knobs such as `sqlite_db`, `faiss_dir`, `embedding_model`, and `rebuild`.
- `retrievers`: typed retriever declarations (`hp_rag`, `vector`) which pre-populate LLM and embedding settings and automatically flip on the vector baseline when needed.
- `hp_rag`: direct overrides for `HyperlinkRetrieverConfig` (e.g. custom prompts, limits).
- `evaluation`: suite identifier plus `questions`, `output`, and optional table rendering hints.

## Defining a Dataset Config

Each dataset config keeps ingestion logic declarative:

```yaml
name: beir-fiqa
data_dir: ./datasets
download:
  url: https://public.ukp…/fiqa.zip
extract:
  archive_type: zip
corpus_relative_path: fiqa/corpus.jsonl
questions_relative_path: fiqa/queries.jsonl
adapter: beir
adapter_kwargs:
  split: dev
```

When referenced by an experiment, the loader resolves relative paths against the config file, pulls the archive (if missing), extracts it into `./datasets/<name>/…`, and returns the corpus/questions paths to the CLI.

## Adding New Presets

1. Drop a dataset description under `configs/datasets/` and ensure `adapter` matches one of `adapter_choices()`.
2. Create an experiment file under `configs/experiments/` that references the dataset config, selects retrievers, and points at your evaluation artefacts.
3. Run `uv run python scripts/run_evals.py --config <path>` to execute. Pass extra CLI flags to override individual values without editing the file.

`src/orchestration/config_loader.py` contains the loader utilities, and the models live in `src/models/configs.py`.
