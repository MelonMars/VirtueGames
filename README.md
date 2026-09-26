# VirtueGames experiments

For the current scoring, audit, family-split, and cross-game steering workflow,
start with [Measurement v2](MEASUREMENT_WORKFLOW.md). Existing saved run scores
and vectors predate these checks and need rescoring/re-extraction.

## One command per experiment, with optional activation extraction

Run `python run_integrity.py`, `python run_integrity_variance.py`,
`python run_calibration.py`, or `python calibration_multisample.py` from the project
root with the options below. The original scripts inside the experiment directories
continue to work. Add `--help` to any command to see all settings.

```sh
# Local GGUF (also accepts --backend llama-cpp; this is the default)
python run_integrity.py --backend gguf --model /path/to/model.gguf

# Local Transformers, without extraction
python run_integrity.py --backend transformers --model Qwen/Qwen3-4B --device cuda

# Install once for TransformerLens extraction (Python 3.10+)
python -m pip install -r requirements-activations.txt

# Transformers + extraction in the same experiment run
python run_integrity.py --backend transformers --model Qwen/Qwen3-4B --device cuda --extract-activations --activation-layers 8,16,24 --activation-types resid_post --activation-positions last-prompt-token

# Inside an existing Runpod PyTorch GPU Pod
bash runpod_setup.sh --activations
python run_integrity_variance.py --runtime runpod --backend transformers --model Qwen/Qwen3-8B --extract-activations --activation-layers last --activation-positions response --out /workspace/results/integrity
```

`--runtime runpod` requires `RUNPOD_POD_ID` (set by Runpod) and selects CUDA for
Transformers. It runs **inside** the Pod; it does not upload the project, rent a GPU,
or forward a local command to Runpod. Follow the setup section below first.
GGUF can run inside a Pod too, using a CUDA-enabled llama-cpp installation.
TransformerLens extraction requires the Transformers backend, not GGUF.

| Option | Values | Default when extraction is enabled |
| --- | --- | --- |
| `--extract-activations` | Include to enable; omit to disable | Disabled |
| `--activation-layers` | `last`, `all`, or zero-based indices such as `0,8,16` | `last` |
| `--activation-types` | Comma-separated `resid_pre,resid_post,attn_out,mlp_out` | `resid_post` |
| `--activation-positions` | `last-prompt-token`, `last-token`, `prompt`, `response`, `all`, `answer-tokens` with optional signed offsets, or indices such as `0,10,-1` | `last-prompt-token` |

Positions are zero-based in the **exact chat-formatted prompt + generated token
sequence**, including special tokens. Negative positions count from its end
(use `--activation-positions=-2,-1` for negative lists). `response` includes all
generated tokens, including EOS when emitted. Layer indices must exist in your
model. Unsupported architectures or missing hook points fail explicitly.

To capture the committed Yes/No token, use `--activation-positions answer-tokens`.
This follows the last-Yes/No-token heuristic: scan only generated tokens and take
the last token whose individual decoded text, stripped and lowercased, is exactly
`yes` or `no`. It does not require a valid `Answer:` line, and does not match
multi-token spellings or tokens that also contain punctuation. If no token matches,
the index records `answer_token_index: null`, `skip_reason: "answer_token_not_found"`,
empty positions and no tensor file; the experiment continues without a fallback.

Relative selections can be combined in order:

```sh
--activation-positions answer-tokens                        # the answer token
--activation-positions answer-tokens-1                      # immediately before it
--activation-positions answer-tokens-1,answer-tokens,answer-tokens+1
```

Offsets count tokens in the exact prompt plus generated sequence. They can reach
back into the prompt or forward to EOS, but cannot go beyond the recorded sequence;
out-of-range offsets fail explicitly. Duplicate positions are removed. Answer-relative
selections cannot be mixed with numeric or other named selections in one flag.
The index records the anchor in `answer_token_index` and actual selected `positions`.
The state at `answer-tokens-1` predicts the answer token; the state at `answer-tokens`
has already processed it and predicts the next token. Extraction replays only the
prefix through the furthest selected position, while saving the full generated IDs.

Each generated response is followed by a teacher-forced extraction pass using its
exact token IDs. Prompt-only selections replay only the required prefix. Selected
positions are copied to CPU inside the hooks; all other activation tensors are
discarded. No second model copy is loaded. This adds compute time, and long
sequences or `all` selections can still require significant memory and disk space.

The run directory contains `activations/call-000000.safetensors`, etc., and
`activations/index.jsonl`. Each index record has the question ID, condition,
sample index, seed, exact token IDs, prompt length, selected positions, hook names,
tensor shapes and file name. Tensors have shape `[1, selected_positions, width]`.
Join the index to result records by question ID, condition and sample; integrity
variance has no treatment captures for questions that fail its control check.
An empty generated suffix produces an index record with no tensor file for a
`response` selection. Failed extraction marks the experiment failed through the
existing status file mechanism.

```python
from safetensors.torch import load_file
cache = load_file("runs_integrity/run-00/activations/call-000000.safetensors")
for hook_name, activation in cache.items():
    print(hook_name, activation.shape)
```

An activation at position t contributes to predicting token t+1. Replay captures
the recorded sequence, with no sampling or interventions during extraction;
floating-point differences from cached generation can occur. Extraction preserves
the model's original weight parameterization. The optional dependency pins
TransformerLens 3.9.0 and requires a compatible Transformers 5.x installation.
TransformerLens uses eager attention for its hooks; enabling extraction may change
generation numerics and memory use compared with the default Transformers attention
kernel. The index records library versions, actual dtype/device, attention implementation
and resolved checkpoint revision when available. Pin the same settings when comparing runs.

The calibration and integrity experiments support two optional runtimes:

- `--backend llama-cpp` (default): an existing local GGUF model using `llama-cpp-python`.
- `--backend transformers`: a Hugging Face causal language model loaded into PyTorch, locally or inside a Runpod GPU Pod. The loaded weights are accessible in Python.

## Local usage

For GGUF, install `llama-cpp-python` (with the appropriate GPU build if needed) and `tqdm`. Existing commands still work:

```sh
python integrity/run_integrity.py --model /path/to/model.gguf --questions /path/to/questions.json
```

For Transformers, use Python 3.10+ and install the optional dependencies. Install a CUDA-enabled PyTorch build for your machine when using a GPU.

```sh
python -m pip install -r requirements-transformers.txt
python integrity/run_integrity.py --backend transformers --model Qwen/Qwen3-4B --device cuda --dtype bfloat16 --questions /path/to/questions.json --out runs_integrity
```

`--model` accepts a Hugging Face model ID or a local checkpoint directory. Use a text instruction/chat model with a chat template supporting system and user messages. Custom remote Python code is disabled. Gated models require accepting the model's terms and authenticating with `hf auth login` or setting `HF_TOKEN` in the environment; tokens are not recorded in experiment configuration.

The same backend flags work with all four entry points:

| Script | Experiment |
| --- | --- |
| `calibration/run_calibration.py` | Single-response calibration |
| `calibration/calibration_multisample.py` | Calibration variance |
| `integrity/run_integrity.py` | Single-response integrity |
| `integrity/run_integrity_variance.py` | Integrity variance |

Transformers options: `--device auto|cpu|cuda`, `--dtype auto|float32|float16|bfloat16`, and `--revision COMMIT_OR_TAG`. Auto selects CUDA when available, otherwise CPU; auto precision is BF16 on compatible CUDA GPUs, FP16 on other CUDA GPUs, and FP32 on CPU. Pin a model commit with `--revision` and save `pip freeze` with your results for reproducibility. `--n-gpu-layers` applies only to llama-cpp.

`--n-ctx` bounds prompt plus generated tokens (also capped by the model's declared context limit). Oversized requests fail rather than silently truncate questions. `--max-tokens` limits newly generated tokens, including reasoning tokens. Reasoning models can need a larger budget to reach the experiment's final answer format.

Temperature zero uses greedy decoding. Variance sampling uses the existing per-sample seeds, top-k 40 and top-p 0.95. Calls start with fresh chat context; seeded calls restore the surrounding PyTorch RNG state. Matching seeds do not imply identical outputs across backends, devices, library versions, or quantizations.

## Optional Runpod workflow

Runpod is the execution location for the Transformers backend. This workflow runs scripts inside a GPU Pod with direct weight access; it does not create a hosted chat API or provision paid resources automatically.

1. Create a GPU Pod using a Runpod PyTorch template with SSH or Jupyter terminal access. A single 24 GB GPU is a starting point for 4B/8B inference in 16-bit precision; extensive activation capture needs more memory. Configure enough disk for the checkpoint, package cache and results (50 GB or more is a useful starting point).
2. Upload this project and your question JSON to `/workspace/VirtueGames` using Jupyter or the Pod's SSH connection instructions. Include the shared root Python files and both experiment directories. Supply `--questions` explicitly for your own dataset, or omit it to use the bundled `calibration/questions.json`.
3. In the Pod terminal:

```sh
cd /workspace/VirtueGames
bash runpod_setup.sh
export HF_HOME=/workspace/.cache/huggingface
python integrity/run_integrity_variance.py \
  --backend transformers --model Qwen/Qwen3-8B \
  --device cuda --dtype bfloat16 \
  --questions /workspace/questions.json \
  --n-samples 8 --temperature 0.7 --seed 0 \
  --n-ctx 4096 --max-tokens 1024 \
  --out /workspace/results/integrity_variance
```

4. Download the output run directory, including `config.json`, `status.json`, result JSONL and summary files. Record dependencies with `python -m pip freeze > /workspace/results/environment.txt`.
5. Stop the Pod when finished. Storage can remain billable while stopped; download results before terminating a Pod or deleting storage. Use a network volume if data must survive Pod termination. Current scripts create new runs; they do not resume interrupted experiments automatically.

See [Runpod Pod connections](https://docs.runpod.io/pods/connect-to-a-pod) and [storage options](https://docs.runpod.io/pods/storage/types).

## Faster integrity multisample runs

The integrity variance runner accepts `--sample-batch-size 4` to generate four
treatment samples concurrently in one model. Start with 2 or 4 and increase only
if peak VRAM allows. Controls remain sequential; activation replay remains one
sequence at a time to bound memory. This is fixed batching within each question,
not continuous batching across questions. Batch size 1 retains the original
per-sample seed policy. Larger batches use `base_seed + first_sample_index` as a
shared batch seed, including the final partial batch. Changing batch size changes
sampled outputs; keep it fixed for comparisons. Each activation record retains
its own sample index and stores the actual shared batch seed.

`--thinking auto` preserves the model template default. Qwen3-4B defaults to
thinking mode; `--thinking off` passes `enable_thinking=False` to the chat template
and can substantially reduce generated text. This changes the experimental
condition, not just execution speed. Other model templates may ignore this option.
The flag is saved in config.json. Qwen recommends avoiding greedy decoding in
thinking mode, whereas integrity controls intentionally use greedy decoding:
see https://huggingface.co/Qwen/Qwen3-4B for the model's guidance.

Example for a **non-thinking** run after uploading the updated source:

```sh
python integrity/run_integrity_variance.py \
  --runtime runpod --backend transformers --model Qwen/Qwen3-4B \
  --device cuda --dtype bfloat16 --thinking off \
  --n-samples 8 --sample-batch-size 4 --temperature 0.7 --seed 0 \
  --n-ctx 4096 --max-tokens 256 \
  --extract-activations --activation-layers last \
  --activation-types resid_post --activation-positions last-prompt-token \
  --out /workspace/results/integrity_fast
```

Pilot on a small question file before launching the full dataset. The progress
bar reports aggregate generation `tok_s` (excluding replay) and `capped` responses
that reached the token budget without EOS. Inspect answer parsing and capped
counts before accepting a shorter output budget. summary.json now includes
generation seconds, extraction seconds, generated tokens, responses and token
limit hits. Compare end-to-end elapsed time for the same questions/settings;
no GPU speedup has been measured in this repository's CPU-only test environment.
Omit extraction flags if activations are unnecessary. Use `--thinking auto` and
the original token budget if preserving the original thinking condition matters.

## Inspecting weights

With the Transformers backend, `llm.model` is the actual PyTorch model and `llm.tokenizer` is its tokenizer. You can inspect named parameters or attach forward hooks before calling an experiment's `run` function. For example:

```python
from pathlib import Path
from config import RunConfig
from llm import load_model, generate

config = RunConfig(model="Qwen/Qwen3-4B", questions=Path("questions.json"),
                   backend="transformers", device="cuda", dtype="bfloat16")
with load_model(config) as llm:
    name, weight = next(llm.model.named_parameters())
    print(name, tuple(weight.shape), weight.dtype)
    print(weight.detach().flatten()[:8].float().cpu())
    print(generate(llm, "Answer honestly.", "Is Paris in France?"))
```

Weights are loaded without added quantization. Use `--extract-activations` for automatic recording, or attach your own Python hooks. Generation and extraction use inference mode, so gradient-based work needs a separate forward pass outside `generate`.

## Verification

```sh
python -m unittest discover -s tests -v
```

The backend integration tests construct a tiny random model and tokenizer locally, without downloads or paid GPU use. They skip when optional Transformers dependencies are absent.
