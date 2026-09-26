from dataclasses import asdict, dataclass
from pathlib import Path

from runs import utc_now


def parse_config(argv=None, *, max_tokens=2048, out="runs", variance=False,
                 calibration=False):
    """Shared options for the individual experiment scripts."""
    import argparse
    from questions import DEFAULT_QUESTIONS
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="GGUF file, or Transformers model ID/directory")
    parser.add_argument("--backend", choices=("llama-cpp", "gguf", "transformers"), default="llama-cpp")
    parser.add_argument("--runtime", choices=("local", "runpod"), default="local",
                        help="Run on this machine, or inside an existing Runpod GPU Pod")
    parser.add_argument("--extract-activations", action="store_true",
                        help="Record TransformerLens activations alongside experiment results (Transformers only)")
    parser.add_argument("--activation-layers", default="last", help="Zero-based layers: last, all, or comma-separated indices")
    parser.add_argument("--activation-types", default="resid_post", help="Comma-separated resid_pre,resid_post,attn_out,mlp_out")
    parser.add_argument("--activation-positions", default="last-prompt-token",
                        help="last-prompt-token, last-token, prompt, response, all, numeric indices, or answer-tokens with signed offsets (e.g. answer-tokens-1,answer-tokens)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--revision", default="main", help="Transformers checkpoint revision (use a commit for reproducibility)")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--out", type=Path, default=Path(out))
    parser.add_argument("--difficulty", default="")
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=max_tokens)
    parser.add_argument("--thinking", choices=("auto", "on", "off"), default="auto",
                        help="Transformers chat-template thinking mode; off changes the experiment behavior")
    if variance:
        if not calibration:
            parser.add_argument("--sample-batch-size", type=int, default=1,
                                help="Integrity treatment samples per Transformers generation call")
        parser.add_argument("--n-samples", type=int, default=8)
        parser.add_argument("--temperature", type=float, default=0.7)
        parser.add_argument("--seed", type=int, default=0)
        if calibration:
            parser.add_argument("--conf-std-threshold", type=float, default=5.0)
    args = vars(parser.parse_args(argv))
    output = args.pop("out")
    args["difficulty"] = tuple(sorted({d.strip() for d in args["difficulty"].split(",") if d.strip()}))
    try:
        if args["backend"] == "gguf":
            args["backend"] = "llama-cpp"
        if args["backend"] == "llama-cpp":
            args["model"] = Path(args["model"])
            if not args["model"].is_file():
                raise ValueError(f"model file does not exist: {args['model']}")
        return RunConfig(**args), output
    except ValueError as exc:
        parser.error(str(exc))


@dataclass(frozen=True)
class RunConfig:
    model: Path | str
    questions: Path
    difficulty: tuple[str, ...] = ()
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    max_tokens: int = 2048
    n_samples: int = 8
    temperature: float = 0.7
    seed: int = 0
    conf_std_threshold: float = 5.0
    backend: str = "llama-cpp"
    device: str = "auto"
    dtype: str = "auto"
    revision: str = "main"
    runtime: str = "local"
    extract_activations: bool = False
    activation_layers: str = "last"
    activation_types: str = "resid_post"
    activation_positions: str = "last-prompt-token"
    thinking: str = "auto"
    sample_batch_size: int = 1

    def __post_init__(self):
        if self.thinking not in ("auto", "on", "off"):
            raise ValueError("unknown thinking mode")
        if self.sample_batch_size < 1:
            raise ValueError("sample_batch_size must be positive")
        if self.backend != "transformers" and (self.thinking != "auto" or self.sample_batch_size != 1):
            raise ValueError("thinking and sample batching require the Transformers backend")
        if self.runtime not in ("local", "runpod"):
            raise ValueError("unknown runtime")
        if self.extract_activations and self.backend != "transformers":
            raise ValueError("--extract-activations requires --backend transformers; GGUF extraction is not supported")
        from activations import validate_selection
        validate_selection(self.activation_layers, self.activation_types, self.activation_positions)
        if self.backend not in ("llama-cpp", "transformers"):
            raise ValueError("unknown backend")
        if self.device not in ("auto", "cpu", "cuda") or self.dtype not in ("auto", "float32", "float16", "bfloat16"):
            raise ValueError("invalid device or dtype")
        if not str(self.model).strip():
            raise ValueError("model must not be empty")
        if min(self.n_ctx, self.max_tokens, self.n_samples) <= 0:
            raise ValueError("n_ctx, max_tokens, and n_samples must be positive")
        import math
        if (not math.isfinite(self.temperature) or self.temperature < 0
                or not math.isfinite(self.conf_std_threshold) or self.conf_std_threshold <= 0):
            raise ValueError("temperature must be finite and nonnegative; confidence threshold must be finite and positive")

    def metadata(self, mode, count):
        values = asdict(self)
        values["model"] = (str(Path(self.model).resolve())
                           if self.backend == "llama-cpp" or Path(self.model).is_dir()
                           else str(self.model))
        values["questions_file"] = str(Path(values.pop("questions")).resolve())
        values["difficulty_filter"] = list(values.pop("difficulty")) or None
        values["base_seed"] = values.pop("seed")
        values.update(mode=mode, n_questions=count, started=utc_now(),
                      reset_before_sample=mode == "calibration-variance")
        if not mode.endswith("variance"):
            values["temperature"] = 0.0
        else:
            values["seed_policy"] = "base_seed + sample_index (reused per question)"
        if mode == "integrity-variance":
            values["control_temperature"] = 0.0
            if self.sample_batch_size > 1:
                values["seed_policy"] = "base_seed + first sample index of each batch; shared RNG within batch (reused per question)"
        return values
