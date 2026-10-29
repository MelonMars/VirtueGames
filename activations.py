"""Optional TransformerLens recording; importing this module needs no ML packages."""
import json
import re
from functools import partial
from importlib.metadata import version
from pathlib import Path

TYPES = ("resid_pre", "resid_post", "attn_out", "mlp_out")
POSITIONS = ("last-prompt-token", "first-response", "last-token", "prompt", "response", "all",
             "confidence-tokens")
ANSWER_POSITION_RE = re.compile(r"answer-tokens([+-]\d+)?")
CONFIDENCE_PATTERNS = {
    "answer": re.compile(r"(?:^|\n)[ \t]*(?:Answer confidence|Confidence):[ \t]*(\d{1,3})%?[ \t]*$", re.I | re.M),
    "decision": re.compile(r"(?:^|\n)[ \t]*Decision confidence:[ \t]*(\d{1,3})%?[ \t]*$", re.I | re.M),
}


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


def find_answer_token_index(tokenizer, full_ids, prompt_len, *, structured_choices=None):
    """Locate a committed token; preserve the legacy locator unless opted into strict lines.

    The structured path selects the first token overlapping the final answer word,
    including multi-token answers such as Unknown. No prose-token fallback is used.
    """
    if structured_choices:
        suffix = [int(token) for token in full_ids[prompt_len:]]
        text = tokenizer.decode(suffix, skip_special_tokens=True)
        choices = '|'.join(re.escape(choice) for choice in structured_choices)
        matches = list(re.finditer(
            r'(?:^|\n)[ \t]*Answer:[ \t]*(' + choices +
            r')[ \t]*\.?[ \t]*(?=\r?$|\r?\n)', text, re.I | re.M))
        if not matches:
            return None
        match = matches[-1]
        start = match.start(1)
        # Work backward: normally only the answer and trailing EOS need decoding.
        for end in range(len(suffix), 0, -1):
            prefix = tokenizer.decode(suffix[:end - 1], skip_special_tokens=True)
            if len(prefix) <= start:
                through = tokenizer.decode(suffix[:end], skip_special_tokens=True)
                if text.startswith(prefix) and len(through) > start and text.startswith(through):
                    return prompt_len + end - 1
                return None
        return None
    hit = None
    for index in range(prompt_len, len(full_ids)):
        piece = tokenizer.decode([int(full_ids[index])]).strip().lower()
        if piece in ("yes", "no"):
            hit = index
    return hit


def find_confidence_token_index(tokenizer, full_ids, prompt_len, field):
    """Find the first token containing the numeric value of a confidence field."""
    suffix = [int(token) for token in full_ids[prompt_len:]]
    text = tokenizer.decode(suffix, skip_special_tokens=True)
    matches = list(CONFIDENCE_PATTERNS[field].finditer(text))
    if not matches:
        return None
    start = matches[-1].start(1)
    for end in range(1, len(suffix) + 1):
        through = tokenizer.decode(suffix[:end], skip_special_tokens=True)
        if len(through) > start and text.startswith(through):
            return prompt_len + end - 1
    return None


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


def select_positions(spec, prompt_length, total_length, *, answer_token_index=None,
                     confidence_token_indices=None):
    if spec == "confidence-tokens":
        return [index for index in (confidence_token_indices or [])
                if index is not None and 0 <= index < total_length]
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
    if spec == "first-response":
        return [prompt_length] if prompt_length < total_length else []
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
        answer_token_index = (find_answer_token_index(self.tokenizer, token_ids, prompt_length,
                              structured_choices=(context or {}).get('structured_answer_choices'))
                              if answer_relative else None)
        confidence_token_indices = (
            [find_confidence_token_index(self.tokenizer, token_ids, prompt_length, field)
             for field in ('answer', 'decision')]
            if self.config.activation_positions == "confidence-tokens" else None)
        positions = select_positions(self.config.activation_positions, prompt_length, tokens.shape[1],
                                     answer_token_index=answer_token_index,
                                     confidence_token_indices=confidence_token_indices)
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
        if confidence_token_indices is not None:
            record.update(position_selection=self.config.activation_positions,
                          confidence_token_indices=confidence_token_indices,
                          skip_reason=("confidence_fields_not_found"
                                       if not any(i is not None for i in confidence_token_indices) else None))
        with (self.directory / "index.jsonl").open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        self.call_index += 1
