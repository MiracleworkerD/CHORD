# CHORD: Cross-Model Hallucination Detection via Relational Graph Discrimination

Reproduction guide for the CHORD experiments.

> **Abstract.**  
> Hallucination detection is critical for deploying large language models (LLMs) in real-world applications. Due to strong empirical performance, internal representation–based methods have emerged as the prevailing direction for detecting hallucinations, yet they remain largely model-specific and often fail to generalize to unseen LLMs. In this paper, we study an important yet underexplored problem, termed *cross-model hallucination detection* (CMHD), which aims to train hallucination detectors on source LLMs while ensuring robust performance on unseen target LLMs. The core challenge of CMHD lies in cross-model representation heterogeneity: hidden states from different LLMs exhibit severe semantic inconsistency, which limits the transferability of classical representation-based detectors. Through analysis and empirical validation, we show that relational structures over layers, tokens, and features capture transferable detection signals despite semantic inconsistency. Based on this observation, we propose **CHORD**: **c**ross-model **h**allucination detection via heterogeneity-**o**riented **r**elational **d**iscrimination, a relational graph discrimination framework that represents hidden states as joint relational graphs over layers, tokens, and features. It feeds these graphs into relation-aware attention to obtain structure-aware features, and uses meta-learning to shape these features toward transferable detection signals. Extensive experiments show that CHORD outperforms representative detection methods in cross-model generalization.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [What Problem Does CHORD Solve?](#what-problem-does-chord-solve)
- [Method Summary](#method-summary)
- [Hardware and VRAM Requirements](#hardware-and-vram-requirements)
- [Environment Setup](#environment-setup)
  - [Install with `uv`](#install-with-uv)
  - [Install PyTorch](#install-pytorch)
  - [Install FlashAttention](#install-flashattention)
  - [Minimal Python Dependencies](#minimal-python-dependencies)
- [Data Pipeline Overview](#data-pipeline-overview)
- [Step 1 — Convert Raw Datasets into the Universal Schema](#step-1-convert-raw-datasets-into-the-universal-schema)
  - [Universal Schema](#universal-schema)
  - [Supported Dataset Adapters](#supported-dataset-adapters)
  - [Example Commands](#example-commands)
  - [Output of Step 1](#output-of-step-1)
- [Step 2 — Generate Answers and Extract Hidden States](#step-2-generate-answers-and-extract-hidden-states)
  - [Why Hidden-State Extraction Is Expensive](#why-hidden-state-extraction-is-expensive)
  - [Expected HDF5 Layout](#expected-hdf5-layout)
  - [Token and Layer Selection](#token-and-layer-selection)
- [Step 3 — Score Generations with Hallucination Labels](#step-3-score-generations-with-hallucination-labels)
- [Step 4 — Train CHORD](#step-4-train-chord)
  - [Expected Training Directory Layout](#expected-training-directory-layout)
  - [Configure Source LLM Domains](#configure-source-llm-domains)
  - [Run Training](#run-training)
  - [Run Evaluation](#run-evaluation)
- [Configuration Reference](#configuration-reference)
  - [DataConfig](#dataconfig)
  - [ModelConfig](#modelconfig)
  - [TrainConfig](#trainconfig)
- [How the Code Maps to the Paper](#how-the-code-maps-to-the-paper)
- [Output Files](#output-files)
- [Reproducibility Notes](#reproducibility-notes)
- [Tips and Troubleshooting](#tips-and-troubleshooting)
- [Citation](#citation)

---

## Overview

This repository contains the implementation and reproduction pipeline for **CHORD**, a cross-model hallucination detector based on relational graph discrimination.

The high-level pipeline has two stages:

1. **Data construction stage**
   - Convert heterogeneous benchmark datasets into a unified schema.
   - Prompt an LLM to generate answers.
   - Extract selected hidden states from the LLM during generation.
   - Use an LLM judge or a dataset-specific evaluator to assign hallucination labels.
   - Save hidden states and metadata in a stable disk format.

2. **Detector training stage**
   - Load hidden states from multiple source LLMs.
   - Restructure hidden states into a shared layer–token–feature representation.
   - Build a relational graph whose nodes are layer–token–feature blocks.
   - Encode the graph with relation-aware attention.
   - Train with meta-learning by holding out source LLMs as pseudo-unseen models.
   - Evaluate cross-model transfer to unseen target LLMs.

The core detector does **not** assume that raw hidden dimensions, layer semantics, or coordinate-level meanings are aligned across LLMs. Instead, it relies on **relational structure**: how selected layers, generated tokens, and feature blocks relate to each other within a model response.

---

## Repository Structure

The released repository uses a flat layout. The main files are:

```text
CHORD/
├── __init__.py
├── config.py
├── data.py
├── evaluate.py
├── generate_pipeline.py
├── hidden_state.py
├── layers.py
├── llm_judge.py
├── metrics.py
├── model.py
├── prepare_datasets.py
├── prompt_builder.py
├── readme.md
├── requirements.txt
├── run_batch.py
├── run_train.py
├── train_meta.py
└── utils.py
```

Only the detector-training files are required once hidden states and labels have already been generated. The dataset preparation, generation, hidden-state extraction, and judging files are needed for reproducing the full benchmark construction pipeline from raw datasets.

The most important files are:

| File | Purpose |
| --- | --- |
| `config.py` | Central configuration file. Paths, graph sizes, model dimensions, and training hyperparameters are defined here. |
| `data.py` | Loads one LLM domain at a time from `02_hidden_states.h5` and `03_final_scored_metadata.jsonl`. Preserves token and layer dimensions. |
| `prepare_datasets.py` | Converts raw benchmark datasets into the universal JSONL schema. |
| `generate_pipeline.py` | Runs LLM generation and coordinates hidden-state extraction for processed examples. |
| `hidden_state.py` | Extracts and writes selected hidden states during generation. |
| `prompt_builder.py` | Builds dataset-specific prompts from universal-schema examples. |
| `llm_judge.py` | Assigns hallucination labels using an LLM judge or task-specific evaluation logic. |
| `layers.py` | Implements scattered padding, edge relation bias, relation-aware self-attention, encoder layers, and attention pooling. |
| `model.py` | Implements the full CHORD detector: hidden states → layer-token-feature nodes → relation-aware encoder → graph embedding → classifier. |
| `train_meta.py` | Implements Feature-Critic-style episodic meta-learning over source LLM domains. |
| `run_train.py` | Main entry point for training. |
| `evaluate.py` | Main entry point for validation and evaluation. |
| `run_batch.py` | Convenience script for running multiple processing or training jobs. |

---

## What Problem Does CHORD Solve?

Standard representation-based hallucination detectors usually assume that the train and test hidden states come from the same LLM, or at least from models with compatible representation geometry. This assumption is fragile.

For example, a probe trained on the hidden states of one model may perform well on that model but collapse when directly applied to another model. The reason is that hidden states are not generally semantically aligned across LLM families, scales, or training recipes. A coordinate that is informative in one model does not necessarily have the same meaning in another model.

CHORD addresses this by changing the object being modeled:

```text
Classical probe:
    raw hidden vector -> classifier

CHORD:
    layer-token-feature relational graph -> relation-aware graph encoder -> classifier
```

Instead of asking whether one hidden coordinate has the same meaning across models, CHORD asks whether **relations among hidden-state substructures** are stable enough to transfer. The method is motivated by the empirical observation that relational patterns over layers, generated tokens, and feature blocks can remain useful even when raw hidden representations are heterogeneous.

---

## Method Summary

Given a question–answer pair and an LLM, we extract hidden states for selected generated tokens and selected transformer layers. For each sample, the raw hidden state tensor has the conceptual form:

```text
[num_generated_tokens, num_layers, hidden_dim]
```

CHORD then performs the following operations.

### 1. Select the final generated tokens and final hidden layers

The current implementation uses the last `K` generated tokens and the last `L` layers by default:

```text
selected hidden states:
    [K, L, hidden_dim]
```

This choice is configurable in `DataConfig`:

```python
last_k_tokens = 2
last_l_layers = 3
```

The last generated tokens are useful because hallucination signals often become clearer near the answer tail, where the model has committed to concrete content. The last layers are useful because deeper layers tend to encode more task-level and semantic information.

### 2. Map heterogeneous hidden dimensions to a shared dimension

Different LLMs can have different hidden sizes:

```text
LLM A: hidden_dim = 4096
LLM B: hidden_dim = 5120
LLM C: hidden_dim = 8192
```

CHORD maps each hidden vector into a shared dimension `max_dim`, usually:

```python
max_dim = 8192
```

The implementation uses **scattered padding** by default, rather than only appending zeros on the right. This distributes the original coordinates over the shared dimension, reducing the positional bias that would occur if all smaller models occupied only the prefix of the padded vector.

The relevant module is:

```text
layers.py
└── ScatteredPadder
```

### 3. Partition the shared feature dimension into blocks

After padding, each hidden vector is partitioned into `B` contiguous blocks:

```python
num_blocks = 64
```

If `max_dim = 8192` and `num_blocks = 64`, then each block has size:

```text
block_size = 8192 / 64 = 128
```

Each selected layer, selected token, and feature block forms one graph node:

```text
node = (layer, token, block)
```

If `last_l_layers = 3`, `last_k_tokens = 2`, and `num_blocks = 64`, each sample has:

```text
3 × 2 × 64 = 384 graph nodes
```

This is the core representation of CHORD.

### 4. Build node representations

Each node contains:

```text
block content
+ layer embedding
+ token embedding
+ block embedding
```

The block content is projected into the model dimension:

```python
model_dim = 256
```

The layer, token, and block embeddings tell the detector where the node came from in the layer–token–feature structure.

The relevant module is:

```text
model.py
└── RelationalHallucinationDetector.encode_nodes
```

### 5. Build relation-aware edge bias

CHORD models pairwise relations between nodes. For a pair of nodes `(u, v)`, the implementation constructs an edge representation from:

```text
relative layer position
relative token position
relative block position
content cosine similarity
content distance
```

Rather than explicitly storing a full edge tensor of shape:

```text
[batch, nodes, nodes, model_dim]
```

the implementation maps the edge features into a multi-head attention bias:

```text
[batch, num_heads, nodes, nodes]
```

This is more memory efficient and directly modulates attention scores:

```text
attention_score(u, v) = Q_u K_v / sqrt(d_head) + relation_bias(u, v)
```

The relevant modules are:

```text
layers.py
├── EdgeRelationBias
└── RelationAwareSelfAttention
```

### 6. Encode the relational graph

The graph is fully connected. Every layer-token-feature node can attend to every other node, while the edge relation bias informs the attention mechanism how those nodes are related.

The relation-aware encoder is stacked for `num_encoder_layers` layers:

```python
num_encoder_layers = 2
```

Each layer contains:

```text
LayerNorm
relation-aware multi-head self-attention
residual connection
feed-forward network
residual connection
```

### 7. Pool the graph into a sample-level embedding

After relation-aware message passing, CHORD applies attention pooling over all graph nodes:

```text
[nodes, model_dim] -> [model_dim]
```

This produces one graph-level representation for the entire question–answer pair.

The relevant module is:

```text
layers.py
└── AttentionPooling
```

### 8. Predict hallucination probability

The graph-level embedding is passed into a small MLP classifier:

```text
graph embedding -> MLP -> logit -> sigmoid -> hallucination probability
```

The training label convention is:

| Label | Meaning |
| --- | --- |
| `1` | hallucinated |
| `0` | correct or abstention |

By default, `correct` and `abstention` are both treated as non-hallucinated labels. This is an engineering convention for the released code: the detector output is a hallucination probability. If a paper version uses the opposite convention, convert scores or labels consistently when reporting metrics. The mapping can be changed in `DataConfig`.

### 9. Train with source-LLM holdout meta-learning

CHORD is trained for cross-model generalization. Each meta-learning episode simulates an unseen target LLM by holding out one source LLM as a pseudo-unseen domain and training on the remaining source LLM domain(s).

A typical episode is:

```text
source LLMs:
    LLM_A, LLM_B, LLM_C

episode:
    meta-train on: LLM_A, LLM_C
    held-out pseudo-unseen validation: LLM_B
```

The minimum runnable setup is:

```text
1 source LLM for meta-training
1 source LLM for pseudo-unseen holdout
```

Therefore, at least **2 source LLM domains** are sufficient to run the episodic holdout protocol. Using 3 or more source LLMs is recommended for more stable and meaningful cross-model meta-learning, because each episode can then train on multiple source domains while reserving one domain for pseudo-unseen validation.

The meta-learning strategy follows a **Feature-Critic-style source-LLM holdout procedure**. At each episode, the held-out source LLM provides a pseudo-unseen validation signal, encouraging parameter updates that transfer across models rather than overfitting to one source model. The implementation is in:

```text
train_meta.py
```

---

## Hardware and VRAM Requirements

CHORD has two different memory bottlenecks:

1. **Hidden-state extraction memory**
2. **Detector training memory**

They should not be confused.

### Hidden-state extraction memory

Extracting hidden states from an LLM is significantly more expensive than ordinary text generation. Standard generation mainly needs model weights and KV cache. Hidden-state extraction additionally captures dense intermediate activations for selected layers and generated tokens. These tensors may temporarily remain on GPU before being copied to CPU and written to disk.

Approximate memory requirements for extraction mode:

| Model Size | Pure Inference VRAM | Hidden-State Extraction VRAM | Suggested Setup |
| --- | ---: | ---: | --- |
| 7B / 8B | ~16–24 GB | ~30–40 GB | 1× A100 40GB / 1× A6000 48GB |
| 13B / 14B | ~30–40 GB | ~50–80 GB | 1× A100 80GB or 2× A100 40GB |
| 30B / 35B | ~70 GB | ~90–130 GB | 2× A100 80GB |
| 70B | ~140–160 GB | >250 GB in many settings | 4× H100 80GB recommended |

These are approximate estimates. Actual usage depends on sequence length, batch size, number of captured layers, generation length, tokenizer behavior, and whether the model implementation returns all hidden states or uses forward hooks.

### Detector training memory

Detector training is much cheaper than hidden-state extraction because it only loads saved hidden vectors, not the full LLM. However, the relational graph is fully connected over `L × T × B` nodes. The attention matrix scales quadratically with the number of nodes.

With the default configuration:

```text
last_l_layers = 3
last_k_tokens = 2
num_blocks = 64
nodes = 384
```

The attention map per sample has:

```text
384 × 384 = 147,456 pairwise relations
```

This is manageable on modern GPUs. If you increase the number of layers, tokens, or blocks aggressively, the quadratic attention cost can become the dominant training bottleneck.

---

## Environment Setup

### Install with `uv`

We recommend using [`uv`](https://github.com/astral-sh/uv) because it is fast and produces clean virtual environments.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate
```

For Windows PowerShell:

```powershell
uv venv
.venv\Scripts\Activate.ps1
```

### Install PyTorch

Install a PyTorch build matching your CUDA version. For CUDA 12.1, for example:

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only debugging:

```bash
uv pip install torch torchvision torchaudio
```

CPU-only mode is useful for checking data parsing and model shape correctness, but it is not recommended for real training.

### Install FlashAttention

The hidden-state extraction pipeline can use FlashAttention / `flash_attention_2` during LLM forward passes when the model, Transformers version, CUDA version, and GPU support it. FlashAttention is especially useful for long prompts, large models, and hidden-state extraction workloads where attention memory can become a bottleneck.

Install FlashAttention according to the official Dao-AILab repository and your local CUDA/PyTorch environment:

```text
https://github.com/dao-ailab/flash-attention
```

A typical installation pattern is:

```bash
uv pip install packaging ninja psutil
uv pip install flash-attn --no-build-isolation
```

If compilation uses too much CPU RAM, limit parallel build jobs:

```bash
MAX_JOBS=4 uv pip install flash-attn --no-build-isolation
```

The exact installation command may depend on CUDA version, PyTorch version, GPU architecture, operating system, and the attention kernels supported by the model.

After installation, verify:

```bash
python - <<'PY'
import torch
import flash_attn
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("flash_attn imported successfully")
PY
```

If your LLM loading code uses Hugging Face Transformers, a typical model loading pattern is:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
```

Some models use different attention backend names. If `flash_attention_2` is not accepted, check the model class and Transformers version.

### Minimal Python Dependencies

A minimal dependency list for the detector is:

```text
torch
numpy
scikit-learn
h5py
tqdm
```

For data processing and hidden-state extraction, typical additional dependencies are:

```text
datasets
transformers
accelerate
safetensors
openai
python-dotenv
aiohttp
pandas
```

A practical `requirements.txt` can start with:

```text
numpy
scikit-learn
h5py
tqdm
datasets
transformers
accelerate
safetensors
openai
python-dotenv
pandas
```

Install with:

```bash
uv pip install -r requirements.txt
```

---

## Data Pipeline Overview

The detector expects each LLM-domain directory to contain:

```text
02_hidden_states.h5
03_final_scored_metadata.jsonl
```

These two files are produced by the upstream data pipeline.

The complete data pipeline is:

```text
Raw benchmark dataset
        │
        ▼
Universal Schema conversion
        │
        ├── 01_structured_data.jsonl
        │
        ▼
LLM generation + hidden-state extraction
        │
        ├── 02_hidden_states.h5
        ├── 02_extracted_metadata.jsonl
        └── 02_hidden_states_failed.jsonl
        │
        ▼
LLM judge / evaluator annotation
        │
        └── 03_final_scored_metadata.jsonl
        │
        ▼
CHORD detector training
```

The detector does not directly load raw Hugging Face datasets. It only consumes the hidden states and labels saved by the previous stages.

---

## Step 1 — Convert Raw Datasets into the Universal Schema

Raw datasets use very different field names and task structures. Some are open-domain QA datasets, some are multiple-choice datasets, some are coding datasets, some are mathematical reasoning datasets, and some already contain hallucinated answers.

To make downstream generation and evaluation uniform, each raw example is converted into a `Universal Schema`.

### Universal Schema

Each processed item contains:

```json
{
  "sample_id": "halueval_qa_train_000001",
  "structured_data": {
    "task_type": "qa",
    "system_instruction": "Please answer the question based strictly on the provided context.",
    "context": "...",
    "question": "...",
    "choices": {},
    "ground_truths": ["..."],
    "incorrect_answers": ["..."]
  },
  "original_doc": {
    "...": "raw dataset fields are preserved here"
  }
}
```

The fields are:

| Field | Type | Description |
| --- | --- | --- |
| `sample_id` | string | Stable unique ID used to align metadata and HDF5 hidden states. |
| `structured_data.task_type` | string | Task family, e.g. `qa`, `multiple_choice`, `reasoning`, `coding`, `summarization`, `agent_action`. |
| `structured_data.system_instruction` | string | Task-specific instruction injected into the prompt. |
| `structured_data.context` | string | Background passage, retrieved context, tool list, tests, story, or other task context. |
| `structured_data.question` | string | Main user-facing question or task prompt. |
| `structured_data.choices` | dict | Multiple-choice options, usually `{ "A": "...", "B": "..." }`. Empty for free-form tasks. |
| `structured_data.ground_truths` | list | Acceptable correct answers. May include aliases or canonical solutions. |
| `structured_data.incorrect_answers` | list | Known hallucinated/trap answers when available. |
| `original_doc` | dict | Raw dataset row retained for debugging and auditability. |

The purpose of `ground_truths` and `incorrect_answers` is important. During judging, a generated answer can be compared not only to the correct answers but also to known hallucinated answers. This is especially useful for datasets such as HaluEval and TruthfulQA where incorrect or hallucinated candidates are part of the dataset.

### Supported Dataset Adapters

The adapter registry currently includes:

| Adapter Name | Dataset | Task Type |
| --- | --- | --- |
| `truthful_qa` | TruthfulQA generation split | QA / truthfulness |
| `halueval_qa` | HaluEval QA | QA hallucination |
| `trivia_qa` | TriviaQA | Open-domain QA |
| `coqa` | CoQA first-turn QA | Conversational QA |
| `coqa_multiturn` | CoQA multi-turn variant | Conversational QA |
| `squad_v2` | SQuAD v2 | Extractive QA / unanswerable QA |
| `arc_challenge` | AI2 ARC-Challenge | Multiple-choice science QA |
| `xsum` | XSum | Summarization |
| `gsm8k` | GSM8K | Mathematical reasoning |
| `human_eval` | HumanEval | Code generation |
| `mbpp` | MBPP | Code generation |
| `xlam_agent` | xLAM function calling | Agent/tool-use action prediction |
| `halubench` | HaluBench | Context-grounded hallucination |
| `hotpotqa` | HotpotQA | Multi-hop QA |
| `commonsenseqa` | CommonsenseQA | Commonsense multiple-choice QA |
| `mmlu` | MMLU | Multi-subject multiple-choice QA |
| `belebele` | Belebele | Reading comprehension |
| `ragtruth` | RAGTruth processed | RAG hallucination |
| `theoremqa` | TheoremQA | Theorem-based reasoning |
| `math` | MATH | Mathematical reasoning |

The adapter design is intentionally simple: each adapter only converts one raw row into the universal schema. It does not generate answers, judge answers, or extract hidden states. This keeps data normalization separate from model inference.

### Example Commands

Process a small validation subset of GSM8K:

```bash
python prepare_datasets.py \
  --dataset gsm8k \
  --output_dir ./unified_datasets \
  --split validation \
  --max_samples 500
```

Process HaluEval QA:

```bash
python prepare_datasets.py \
  --dataset halueval_qa \
  --output_dir ./unified_datasets \
  --split data \
  --max_samples 5000
```

Process MBPP:

```bash
python prepare_datasets.py \
  --dataset mbpp \
  --output_dir ./unified_datasets \
  --split train \
  --max_samples 1000
```

Process MMLU:

```bash
python prepare_datasets.py \
  --dataset mmlu \
  --output_dir ./unified_datasets \
  --split test \
  --max_samples 5000
```

If the requested split does not exist, the adapter script automatically probes available splits and falls back to a valid split according to this priority:

```text
train -> validation -> test -> data -> first available split
```

### Output of Step 1

The output file is:

```text
./unified_datasets/{dataset_name}_{split}.jsonl
```

Example:

```text
./unified_datasets/halueval_qa_data.jsonl
```

Each line is a complete JSON object. This line-oriented format is robust for large-scale experiments because it supports streaming, partial recovery, and easy debugging with tools such as `head`, `tail`, and `jq`.

---

## Step 2 — Generate Answers and Extract Hidden States

After schema conversion, the generation script loads a model, builds a task-specific prompt, generates an answer, and extracts hidden states for the generated tokens.

A typical command is:

```bash
python generate_pipeline.py \
  --model_path ./models/Qwen3-14B \
  --input_jsonl ./unified_datasets/halueval_qa_data.jsonl \
  --output_dir ./experiments/Qwen/Qwen3-14B/halueval_qa_5000samples \
  --max_samples 5000 \
  --dtype bf16 \
  --attn_implementation flash_attention_2
```

Depending on your implementation, the exact argument names may differ. The important point is that the output directory must eventually contain:

```text
02_hidden_states.h5
03_final_scored_metadata.jsonl
```

### Why Hidden-State Extraction Is Expensive

Hidden-state extraction is not the same as ordinary generation.

During normal generation, the model only needs to produce logits and maintain KV cache. During extraction, we additionally save internal activations. If we extract many layers or many generated tokens, the activation tensors become very large.

For one token and one layer, the hidden vector has shape:

```text
[hidden_dim]
```

For many generated tokens and many layers, the stored tensor is conceptually:

```text
[num_generated_tokens, num_selected_layers, hidden_dim]
```

For large models, `hidden_dim` can be 8192 or larger. For long generations, the number of generated tokens can also be large. This is why extraction should be done carefully:

- use BF16/FP16 where possible;
- avoid unnecessarily large batch sizes;
- extract only the layers needed for experiments;
- write hidden states to disk incrementally;
- use an HDF5 writer process if GPU generation should not be blocked by disk I/O;
- enable FlashAttention when the model supports it.

### Expected HDF5 Layout

CHORD expects the HDF5 file to be organized by `sample_id`. Each sample contains a group named `generated_tokens`. Each generated token contains one dataset per extracted layer.

```text
02_hidden_states.h5
└── {sample_id}/
    └── generated_tokens/
        ├── token_000/
        │   ├── layer_00
        │   ├── layer_01
        │   ├── ...
        │   └── layer_XX
        ├── token_001/
        │   ├── layer_00
        │   ├── layer_01
        │   ├── ...
        │   └── layer_XX
        └── ...
```

The detector loader reads this hierarchy as:

```python
token_group = f_h5[sample_id]["generated_tokens"]

for token_key in sorted(token_group.keys()):
    layer_group = token_group[token_key]
    for layer_key in sorted(layer_group.keys()):
        hidden = layer_group[layer_key][:]
```

The loader supports hidden vectors stored either as ordinary floating-point arrays or as BF16 values serialized through an integer/byte representation. If BF16 values are stored through an int16/byte buffer, the loader reinterprets the stored buffer as BF16; if hidden states are stored directly as `float16`, `bfloat16`, or `float32` arrays, the loader converts them with `torch.as_tensor`.

The resulting per-sample tensor has conceptual shape:

```text
[num_generated_tokens, num_layers, hidden_dim]
```

### Token and Layer Selection

The loader then selects the final tokens and final layers:

```python
last_k_tokens = 2
last_l_layers = 3
```

If a sample has fewer than the required number of tokens or layers, the loader pads the missing positions with zeros. After selection, every sample from the same LLM domain has shape:

```text
[last_k_tokens, last_l_layers, hidden_dim]
```

During batching, the shape becomes:

```text
[batch_size, last_k_tokens, last_l_layers, hidden_dim]
```

This shape is important. CHORD does **not** average across tokens or layers before graph construction. The token and layer axes are preserved so the model can build relational graphs over them.

---

## Step 3 — Score Generations with Hallucination Labels

After generating answers and extracting hidden states, the pipeline assigns a label to each generated response.

The final metadata file is:

```text
03_final_scored_metadata.jsonl
```

Each line should contain at least:

```json
{
  "sample_id": "halueval_qa_data_000001",
  "eval_category": "hallucination",
  "eval_reasoning": "The answer contradicts the provided context..."
}
```

The detector uses `eval_category` as the label source.

Default mapping:

| `eval_category` | Detector label |
| --- | ---: |
| `hallucination` | `1.0` |
| `correct` | `0.0` |
| `abstention` | `0.0` |
| other / missing | skipped |

The label mapping can be changed in `DataConfig`:

```python
positive_label = "hallucination"
negative_labels = ("correct", "abstention")
```

The `sample_id` in `03_final_scored_metadata.jsonl` must match a top-level group in `02_hidden_states.h5`.

---

## Step 4 — Train CHORD

### Expected Training Directory Layout

Each source LLM/domain directory should look like:

```text
./experiments/
└── Qwen/
    └── Qwen3-14B/
        └── halueval_qa_5000samples/
            ├── 02_hidden_states.h5
            └── 03_final_scored_metadata.jsonl
```

For cross-model training, you need multiple such directories:

```text
./experiments/
├── Qwen/Qwen3-8B/halueval_qa_5000samples/
│   ├── 02_hidden_states.h5
│   └── 03_final_scored_metadata.jsonl
├── Qwen/Qwen3-14B/halueval_qa_5000samples/
│   ├── 02_hidden_states.h5
│   └── 03_final_scored_metadata.jsonl
├── Llama/Llama-3.1-8B/halueval_qa_5000samples/
│   ├── 02_hidden_states.h5
│   └── 03_final_scored_metadata.jsonl
└── Ministral/Ministral-3-14B/halueval_qa_5000samples/
    ├── 02_hidden_states.h5
    └── 03_final_scored_metadata.jsonl
```

### Configure Source LLM Domains

Edit `config.py`:

```python
source_llm_dirs = {
    "qwen3_8b": "./experiments/Qwen/Qwen3-8B/halueval_qa_5000samples",
    "qwen3_14b": "./experiments/Qwen/Qwen3-14B/halueval_qa_5000samples",
    "llama3_8b": "./experiments/Llama/Llama-3.1-8B/halueval_qa_5000samples",
    "ministral_14b": "./experiments/Ministral/Ministral-3-14B/halueval_qa_5000samples",
}
```

At least two source LLMs are required for episodic holdout meta-learning:

```text
1 source LLM for meta-training
1 source LLM for pseudo-unseen validation
```

With only two source LLMs, set:

```python
min_meta_train_domains = 1
```

For stronger experiments, use three or more source LLMs and set `min_meta_train_domains` to `2` if you want each episode to train on at least two non-held-out source domains.

### Run Training

From the repository root:

```bash
python run_train.py
```

During training, the script prints logs similar to:

```text
Ep 00050 | meta_train=qwen3_8b,llama3_8b | heldout=qwen3_14b | hold_auc=0.6812 | val_mean_auc=0.7025
```

The printed values mean:

| Field | Meaning |
| --- | --- |
| `meta_train` | Source LLMs used for inner-loop adaptation in this episode. |
| `heldout` | Source LLM treated as pseudo-unseen for this episode. |
| `hold_auc` | AUC on the held-out pseudo-unseen batch after the inner update. |
| `val_mean_auc` | Mean validation AUC across source domains. |

### Run Evaluation

After training:

```bash
python evaluate.py
```

This loads:

```text
outputs/chord/best.pt
```

and evaluates the detector on the validation split of each configured source domain. If `target_llm_dirs` is configured in `config.py`, evaluation also reports cross-model transfer on those target LLM directories. Source-domain validation is mainly used for checkpoint selection and sanity checking; target-domain evaluation corresponds to the cross-model setting reported in the paper.

---

## Configuration Reference

All main hyperparameters are in:

```text
config.py
```

### DataConfig

| Field | Default | Description |
| --- | ---: | --- |
| `source_llm_dirs` | dict | Mapping from source LLM/domain name to experiment directory. |
| `target_llm_dirs` | dict | Optional target LLM directories for additional evaluation. |
| `hidden_filename` | `02_hidden_states.h5` | HDF5 file containing hidden states. |
| `metadata_filename` | `03_final_scored_metadata.jsonl` | JSONL file containing labels and metadata. |
| `token_group_name` | `generated_tokens` | HDF5 group containing generated-token hidden states. |
| `layer_prefix` | `layer_` | Prefix used to identify layer datasets in HDF5. |
| `sample_id_key` | `sample_id` | Metadata key used to align JSONL and HDF5 entries. |
| `label_key` | `eval_category` | Metadata key used as the hallucination label source. |
| `positive_label` | `hallucination` | Label mapped to `1`. |
| `negative_labels` | `("correct", "abstention")` | Labels mapped to `0`. |
| `last_k_tokens` | `2` | Number of final generated tokens used by the graph. |
| `last_l_layers` | `3` | Number of final layers used by the graph. |
| `val_ratio` | `0.2` | Fraction of each source domain used for validation. |
| `cache_in_memory` | `True` | Whether to load hidden states into memory at startup. |

### ModelConfig

| Field | Default | Description |
| --- | ---: | --- |
| `max_dim` | `8192` | Shared hidden dimension after padding. |
| `num_blocks` | `64` | Number of feature blocks per hidden vector. |
| `model_dim` | `256` | Internal node embedding dimension. |
| `rel_pos_dim` | `48` | Embedding dimension for relative layer/token/block positions. |
| `num_heads` | `8` | Number of attention heads. |
| `num_encoder_layers` | `2` | Number of relation-aware encoder layers. |
| `ff_mult` | `4` | Feed-forward expansion factor. |
| `dropout` | `0.1` | Dropout in relation-aware encoder. |
| `classifier_hidden_dim` | `128` | Hidden dimension of the final MLP classifier. |
| `classifier_dropout` | `0.3` | Dropout in classifier. |
| `use_scattered_padding` | `True` | Whether to use scattered padding rather than prefix padding. |
| `use_content_distance` | `True` | Whether edge bias uses pairwise content distance. |
| `use_content_cosine` | `True` | Whether edge bias uses pairwise cosine similarity. |
| `edge_bias_clip` | `5.0` | Clipping range for learned attention bias. |

### TrainConfig

| Field | Default | Description |
| --- | ---: | --- |
| `seed` | `42` | Random seed. |
| `device` | `cuda` | Training device. Falls back to CPU if CUDA is unavailable. |
| `precision` | `bf16` | Autocast precision. |
| `batch_size` | `32` | Training batch size per domain batch. |
| `eval_batch_size` | `128` | Evaluation batch size. |
| `episodes` | `3000` | Number of meta-learning episodes. |
| `inner_steps` | `2` | Number of inner-loop steps per selected source domain. |
| `min_meta_train_domains` | `1` | Minimum number of non-held-out source LLMs used in each meta-train episode. Set to `2` when using at least three source domains and requiring multi-source meta-training in every episode. |
| `max_meta_train_domains` | `4` | Maximum number of source LLMs used in each meta-train episode. |
| `inner_lr` | `5e-4` | Inner-loop learning rate. |
| `meta_lr` | `0.1` | Outer/meta update rate used by the Feature-Critic-style episodic training procedure. |
| `weight_decay` | `1e-2` | AdamW weight decay. |
| `label_smoothing` | `0.05` | Binary label smoothing factor. |
| `grad_clip_norm` | `1.0` | Gradient clipping norm. |
| `eval_every` | `50` | Evaluation frequency in episodes. |
| `save_every` | `200` | Periodic checkpoint frequency. |
| `output_dir` | `./outputs/chord` | Output directory for checkpoints. |
| `checkpoint_name` | `best.pt` | Best-checkpoint filename. |

---

## How the Code Maps to the Paper

| Paper Component | Code Location | Implementation |
| --- | --- | --- |
| Heterogeneous hidden-state loading | `data.py` | Loads per-LLM hidden states from HDF5. |
| Last-token / last-layer selection | `data.py` | Selects `last_k_tokens` and `last_l_layers`. |
| Shared hidden dimension | `layers.py::ScatteredPadder` | Maps variable hidden dimensions to `max_dim`. |
| Feature block partition | `model.py::encode_nodes` | Splits padded hidden dimension into `num_blocks`. |
| Layer-token-feature graph nodes | `model.py::encode_nodes` | Each node corresponds to one `(layer, token, block)`. |
| Node content projection | `model.py` | `block_projector`. |
| Layer/token/block embeddings | `model.py` | `layer_embed`, `token_embed`, `block_embed`. |
| Relative position edge features | `layers.py::EdgeRelationBias` | Uses relative layer/token/block embeddings. |
| Content relation edge features | `layers.py::EdgeRelationBias` | Uses cosine similarity and distance. |
| Relation-aware attention | `layers.py::RelationAwareSelfAttention` | Adds learned relation bias to attention scores. |
| Graph encoder | `layers.py::RelationAwareEncoderLayer` | Stacked attention + feed-forward blocks. |
| Attention pooling | `layers.py::AttentionPooling` | Learns node weights and pools to a graph embedding. |
| Hallucination classifier | `model.py` | MLP over graph embedding. |
| Meta-learning | `train_meta.py` | Source-domain episodic holdout training. |
| Metrics | `metrics.py` | Accuracy, F1, AUROC, average precision. |

---

## Output Files

The training output directory is:

```text
./outputs/chord/
```

Typical files:

```text
outputs/chord/
├── best.pt
├── episode_200.pt
├── episode_400.pt
└── ...
```

`best.pt` contains:

```python
{
    "model": model_state_dict,
    "episode": best_episode,
    "best_mean_auc": best_mean_auc,
    "evals": validation_metrics,
    "config": config_object
}
```

Periodic checkpoints contain recent training history for debugging and recovery.

---

## Reproducibility Notes

For reproducible results:

1. Fix `seed` in `TrainConfig`.
2. Use the same source LLM directories.
3. Use the same hidden-state extraction settings.
4. Use the same selected token and layer counts.
5. Use the same `max_dim` and `num_blocks`.
6. Use the same label mapping.
7. Avoid mixing hidden states extracted with different prompt templates unless explicitly studying prompt sensitivity.
8. Keep the same precision where possible.
9. Use deterministic dataset splits when comparing methods.

The most common reason for irreproducible results is that the hidden-state extraction pipeline changed while the detector configuration remained the same. For example, changing from last-token extraction to all-token extraction changes the distribution of hidden states even if the detector code is unchanged.

---

## Tips and Troubleshooting

### `RuntimeError: Need at least 2 source LLMs`

Feature-Critic-style source-LLM holdout requires at least two source domains:

```text
one for meta-training + one held-out pseudo-unseen source
```

If you have exactly two source LLMs, set:

```python
min_meta_train_domains = 1
```

If `min_meta_train_domains = 2`, then at least three source LLMs are required because one source domain must still be held out in each episode.

### `FileNotFoundError: missing 02_hidden_states.h5 or 03_final_scored_metadata.jsonl`

The detector only consumes processed experiment directories. Make sure the directory path points to the folder that directly contains:

```text
02_hidden_states.h5
03_final_scored_metadata.jsonl
```

Not the parent folder.

### `sample_id not found in HDF5`

The `sample_id` in the metadata file must match the top-level group name in `02_hidden_states.h5`.

Check with:

```python
import h5py, json

h5_path = "02_hidden_states.h5"
jsonl_path = "03_final_scored_metadata.jsonl"

with h5py.File(h5_path, "r") as f:
    h5_ids = set(f.keys())

with open(jsonl_path, "r", encoding="utf-8") as g:
    meta_ids = {json.loads(line)["sample_id"] for line in g if line.strip()}

print("metadata not in h5:", len(meta_ids - h5_ids))
print("h5 not in metadata:", len(h5_ids - meta_ids))
```

### HDF5 layer keys are sorted incorrectly

Layer and token keys should contain numeric suffixes such as:

```text
token_000
token_001
layer_00
layer_01
```

The loader uses natural integer sorting, so `layer_10` will come after `layer_9`. Avoid arbitrary layer names without numbers.

### BF16 hidden vectors look corrupted

The detector assumes that some HDF5 datasets may store BF16 data through an int16/byte representation. If your extraction pipeline stores normal `float16` or `float32` arrays instead, the loader will convert them through `torch.as_tensor`.

If you changed the storage format, verify one sample manually:

```python
import h5py
import torch

with h5py.File("02_hidden_states.h5", "r") as f:
    sid = list(f.keys())[0]
    tok = list(f[sid]["generated_tokens"].keys())[0]
    layer = list(f[sid]["generated_tokens"][tok].keys())[0]
    x = f[sid]["generated_tokens"][tok][layer][:]
    print(type(x), getattr(x, "shape", None), getattr(x, "dtype", None))
```

### Out of memory during detector training

Reduce one or more of:

```python
batch_size
num_blocks
last_k_tokens
last_l_layers
num_encoder_layers
model_dim
num_heads
```

Remember that attention cost grows quadratically with:

```text
nodes = last_l_layers × last_k_tokens × num_blocks
```

### Out of memory during hidden-state extraction

Reduce:

```text
generation batch size
number of captured layers
number of generated tokens
max_new_tokens
prompt length
```

Also consider:

```text
attn_implementation="flash_attention_2"
device_map="auto"
device_map="balanced_low_0"
torch_dtype=torch.bfloat16
```

For 70B-scale extraction, use multi-GPU inference and conservative batch sizes.

### FlashAttention installation fails

Check:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvcc -V
ninja --version
```

Then install prerequisites:

```bash
uv pip install packaging ninja psutil
```

Try limiting compilation jobs:

```bash
MAX_JOBS=4 uv pip install flash-attn --no-build-isolation
```

If it still fails, consult the official FlashAttention repository because supported CUDA, PyTorch, GPU, and OS combinations change over time.

### Validation AUC is unstable

Possible reasons:

- too few source LLM domains;
- severe label imbalance;
- too small validation split;
- inconsistent generation prompts across LLMs;
- hidden states extracted from different layer sets;
- excessive graph size relative to dataset size;
- source LLMs are too similar, making pseudo-unseen holdout too easy;
- source LLMs are too different, making early meta-training unstable.

Practical fixes:

```python
episodes = 5000
inner_steps = 1 or 2
meta_lr = 0.05
batch_size = 16 or 32
label_smoothing = 0.05
```

### Single-domain sanity check

Before running full meta-learning, it is useful to verify that each domain can be learned at all. You can temporarily configure only one domain and write a standard supervised training loop, or disable the meta-learning requirement for debugging. A single-domain AUC much lower than expected often indicates a label or hidden-state extraction issue.

---

## Citation

This paper has been accepted as a Spotlight presentation at NeurIPS 2026.
If you use this repository, please cite:

```bibtex
@inproceedings{deng2026chord,
  title={CHORD: Cross-Model Hallucination Detection via Relational Graph Discrimination},
  author={Yongxin Deng and Zhen Fang and Guansong Pang and Sharon Li and Ling Chen},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```
