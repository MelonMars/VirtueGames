"""Optional TransformerLens recording; importing this module needs no ML packages."""
import json
import re
from functools import partial
from importlib.metadata import version
from pathlib import Path

TYPES = ("resid_pre", "resid_post", "attn_out", "mlp_out")
POSITIONS = ("last-prompt-token", "last-token", "prompt", "response", "all")
ANSWER_POSITION_RE = re.compile(r"answer-tokens([+-]\d+)?")


def answer_offsets(spec):
    """None for an ordinary selection; offsets for an answer-relative selection."""
    if "answer-tokens" not in spec:
        return None
    result = []
    for part in spec.split(','):
        match = ANSWER_POSITION_RE.fullmatch(part.strip())
        if match is None:
            raise ValueError("Use answer-tokens, answer-tokens-1, or answer-tokens+1 (comma-separated)")
        result.append(int(match.group(1) or 0))
    return list(dict.fromkeys(result))


def find_answer_token_index(tokenizer, full_ids, prompt_len):
    """Last completion token decoding exactly to Yes/No after whitespace/case normalization."""
    hit = None
    for index in range(prompt_len, len(full_ids)):
        piece = tokenizer.decode([int(full_ids[index])]).strip().lower()
        if piece in ("yes", "no"):
            hit = index
    return hit


def indices(value):
    try:
        result = list(dict.fromkeys(int(part.strip()) for part in value.split(',')))
    except ValueError as exc:
        raise ValueError(f"Expected comma-separated integer indices, got {value!r}") from exc
    return result


def validate_selection(layers, types, positions):
    if layers not in ("last", "all") and any(i < 0 for i in indices(layers)):
        raise ValueError("Activation layer indices must be nonnegative")
    if any(t.strip() not in TYPES for t in types.split(',')):
        raise ValueError(f"Activation types must be chosen from {','.join(TYPES)}")
    if positions not in POSITIONS and answer_offsets(positions) is None:
        indices(positions)


def select_positions(spec, prompt_length, total_length, *, answer_token_index=None):
    offsets = answer_offsets(spec)
    if offsets is not None:
        if answer_token_index is None:
            return []
        result = [answer_token_index + offset for offset in offsets]
        if any(p < 0 or p >= total_length for p in result):
            raise ValueError(f"Answer-relative activation position outside sequence of {total_length} tokens: {spec}")
        return result
    if spec == "last-prompt-token":
        return [prompt_length - 1]
    if spec == "last-token":
        return [total_length - 1]
    if spec == "prompt":
        return list(range(prompt_length))
    if spec == "response":
        return list(range(prompt_length, total_length))
    if spec == "all":
        return list(range(total_length))
    result = [p if p >= 0 else total_length + p for p in indices(spec)]
    if any(p < 0 or p >= total_length for p in result):
        raise ValueError(f"Activation position outside sequence of {total_length} tokens: {spec}")
    return list(dict.fromkeys(result))


class ActivationRecorder:
    def __init__(self, model, tokenizer, config):
        try:
            from transformer_lens.model_bridge import TransformerBridge
        except ImportError as exc:
            raise ImportError("Activation extraction requires: pip install -r requirements-activations.txt") from exc
        self.config = config
        self.tokenizer = tokenizer
        self.bridge = TransformerBridge.boot_transformers(
            str(config.model), hf_model=model, tokenizer=tokenizer,
            device=model.device, dtype=model.dtype, revision=config.revision)
        self.bridge.eval()
        self.runtime_metadata = dict(
            resolved_revision=getattr(model.config, '_commit_hash', None),
            dtype=str(model.dtype), device=str(model.device),
            attention_implementation=getattr(model.config, '_attn_implementation', None),
            versions={package: version(package) for package in ('torch', 'transformers', 'transformer-lens')})
        count = self.bridge.cfg.n_layers
        layers = (list(range(count)) if config.activation_layers == "all" else
                  [count - 1] if config.activation_layers == "last" else indices(config.activation_layers))
        if any(layer >= count for layer in layers):
            raise ValueError(f"Model has {count} layers; valid indices are 0 through {count - 1}")
        self.names = [f"blocks.{layer}.hook_{kind}" for layer in layers
                      for kind in dict.fromkeys(t.strip() for t in config.activation_types.split(','))]
        missing = [name for name in self.names if name not in self.bridge.hook_dict]
        if missing:
            raise ValueError(f"Selected hooks unavailable for this architecture: {missing}")
        self.directory = None
        self.call_index = 0

    def begin_run(self, run_dir):
        self.directory = Path(run_dir) / "activations"
        self.directory.mkdir(exist_ok=False)
        self.call_index = 0

    def capture(self, tokens, prompt_length, *, context, seed, temperature):
        import torch
        from safetensors.torch import save_file
        if self.directory is None:
            raise RuntimeError("Call begin_run(run_dir) before generating with activation extraction")
        token_ids = tokens[0].detach().cpu().tolist()
        answer_relative = answer_offsets(self.config.activation_positions) is not None
        answer_token_index = (find_answer_token_index(self.tokenizer, token_ids, prompt_length)
                              if answer_relative else None)
        positions = select_positions(self.config.activation_positions, prompt_length, tokens.shape[1],
                                     answer_token_index=answer_token_index)
        captured = {}

        def save_activation(value, hook, *, name):
            # Clone on CPU so each file contains only the requested positions.
            captured[name] = value[:, positions, :].detach().to('cpu').contiguous().clone()

        if positions:
            # Causality lets prompt-only extraction skip the generated suffix.
            replay = tokens[:, :max(positions) + 1]
            with torch.inference_mode():
                self.bridge.run_with_hooks(replay, fwd_hooks=[(name, partial(save_activation, name=name)) for name in self.names],
                                           return_type=None, use_cache=False)
            if set(captured) != set(self.names):
                raise RuntimeError(f"Expected hooks {self.names}; captured {list(captured)}")
        stem = f"call-{self.call_index:06d}"
        filename = f"{stem}.safetensors" if captured else None
        if filename:
            save_file(captured, str(self.directory / filename))
        record = dict(call_index=self.call_index, context=context or {}, seed=seed,
                      temperature=temperature, prompt_length=prompt_length,
                      token_ids=token_ids, positions=positions,
                      hooks=self.names, file=filename,
                      shapes={name: list(value.shape) for name, value in captured.items()},
                      capture_method="teacher-forced replay", model=str(self.config.model),
                      revision=self.config.revision, runtime=self.runtime_metadata)
        if answer_relative:
            record.update(position_selection=self.config.activation_positions,
                          answer_token_index=answer_token_index,
                          skip_reason="answer_token_not_found" if answer_token_index is None else None)
        with (self.directory / "index.jsonl").open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        self.call_index += 1
