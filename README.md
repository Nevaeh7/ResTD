# Residual Trajectory Distillation for Generative Retrieval

This repository contains the source code for **ResTD**, proposed in the paper
**Residual Trajectory Distillation for Generative Retrieval**.
This release supports ESCI-US, ESCI-ES, and ESCI-JP.

The pipeline has three entry points: **data preparation → tokenization → generative retrieval (GR)**.

## Installation

Use Linux x86_64, Python 3.12, and an NVIDIA A100. Install a CUDA-enabled PyTorch
wheel compatible with your driver, then the pinned training dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0
python -m pip install -r requirements.lock.txt
```

Run commands from this repository. `PYTHON=/path/to/python` can select another
interpreter for the shell entry points. GPU 0 is the default; use
`CUDA_VISIBLE_DEVICES` to select a GPU.
`requirements.lock.txt` fixes the full dependency set for this environment;
`requirements.txt` lists the direct dependencies.

## Training and evaluation

This path trains a new SID index and retriever from public inputs.

```bash
# 1. Download the public ESCI and ESCI-S inputs and prepare all three locales.
bash scripts/01_prepare_data.sh

# 2. Encode items with BERT, train the RQ-VAE, resolve SID collisions,
#    and export residual trajectories along the final stored SIDs.
bash scripts/02_tokenize.sh us

# 3. Train the base retriever, continue with ResTD, and evaluate.
bash scripts/03_gr.sh us
```

Replace `us` with `es` or `jp` in steps 2 and 3. Step 1 prepares all three locales.
To train and evaluate all locales after data preparation:

```bash
for locale in us es jp; do
  bash scripts/02_tokenize.sh "$locale"
  bash scripts/03_gr.sh "$locale"
done
```

Step 3 writes the trained checkpoint and evaluation results under
`outputs/<locale>/seed<seed>/`.

All three commands support `--data-root`; steps 2 and 3 also support
`--output-root`. Existing incompatible files are rejected.

For already downloaded raw inputs, step 1 accepts:

```bash
bash scripts/01_prepare_data.sh \
  --hf-data-dir /path/to/esci/data \
  --esci-s-json-zst /path/to/esci.json.zst
```

A local public model snapshot can be supplied with `--embedding-model` in step 2
and `--base-model` in step 3. To continue an existing trained retriever, pass
`--base-checkpoint /path/to/checkpoint`; this skips base training. Its SID index,
category vocabulary, and residual cache must match the prepared dataset.
An optional `--category-adapter` loads a final-category-head adapter before the
ResTD continuation. `--resume` resumes an interrupted GR phase from its latest
Trainer checkpoint, when available.

### Training configuration

The locale configurations are in `configs/us.json`, `configs/es.json`, and
`configs/jp.json`.

- **Item encoder:** BERT-base-uncased for US; multilingual BERT-base-cased for ES/JP.
- **Indexer:** four 256-entry codebooks, residual dimension 32, a frozen index during GR.
- **Retriever:** T5-base for US; mT5-base for ES/JP; three latent category states.
- **ResTD continuation:** H=4, 1,200 optimizer updates, learning rate 1e-5,
  effective batch 128, microbatch 8, BF16, cosine decay, and 120-update warmup.
- **Distillation:** teacher/student temperature 0.2, horizon decay 0.7,
  coefficient 0.1, adaptive collision correction with floor 0.1 and margin 0.001,
  and a detached cap of 5% of the SID loss. Category heads remain frozen during continuation.
- **Base initialization:** public pretrained weights followed by base retrieval training.
  A query-disjoint split of training queries selects the base checkpoint; test
  queries are not used for this selection.

On memory-constrained GPUs, `--micro-batch-size 4` or `2` increases gradient
accumulation automatically while retaining effective batch 128. Full training is substantially longer than a
short validation run. The three seeds can be run with `--seed 42`, `2027`, and `2028`.

## Evaluation protocol

The default evaluation configuration uses beam 100, category width 3, prefix-constraint weights
2/2/6 for US/ES/JP, lexical weight 0.25, at most two lexical constraints, NFKC
normalization, and measurement constraints. Category expansion and ranking fusion
are not used.

The evaluator treats E/S/C as positives for Recall and
uses gains **3/2/1** for NDCG. It deduplicates complete SIDs and preserves the
fixed index's category ownership. Use `--relevance binary` to evaluate with
binary relevance.

```bash
python -m restd.evaluate --locale us \
  --checkpoint /path/to/checkpoint --data-root data \
  --output outputs/us/binary.json --relevance binary
```

`--decoding standard` selects ordinary category-trie beam search without the
prefix score adjustment. Evaluation checks the expected population of
6,014 US, 1,656 ES, and 1,883 JP queries.

## Layout

```text
configs/                 Locale and evaluation settings
scripts/                 The three user-facing entry points
restd/prepare*.py         ESCI and category preparation
restd/embeddings.py       Item text encoding
restd/rq/                 RQ-VAE and final-SID residual export
restd/modeling.py         T5/mT5 ResTD training objective
restd/train.py            Base training and ResTD continuation
restd/evaluate.py         Final checkpoint retrieval and metrics
restd/constraints.py      Product constraints used by the final decoder
restd/decoding.py         Prefix score adjustment
```

## Data and acknowledgments

The preprocessing and latent-reasoning backbone build on CaLIR / ReGEN.
The data come from Amazon ESCI, the tasksource ESCI export, and ESCI-S.
Model implementations use Hugging Face Transformers.
