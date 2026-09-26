import copy
import gc
import time
from contextlib import nullcontext


class TransformersModel:
    def __init__(self, config):
        self.activation_recorder = None
        self.thinking = config.thinking
        self.performance = dict(generation_seconds=0.0, extraction_seconds=0.0,
                                generated_tokens=0, responses=0, token_limit_hits=0)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install optional dependencies: pip install -r requirements-transformers.txt") from exc
        self.torch = torch
        self.n_ctx = config.n_ctx
        device = config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; install CUDA-enabled PyTorch or use --device cpu")
        dtype = config.dtype
        if dtype == "auto":
            dtype = "float32" if device == "cpu" else (
                "bfloat16" if torch.cuda.is_bf16_supported() else "float16")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(config.model), revision=config.revision, trust_remote_code=False)
        if not self.tokenizer.chat_template:
            raise ValueError("Choose an instruction/chat model with a tokenizer chat template")
        self.model = AutoModelForCausalLM.from_pretrained(
            str(config.model), revision=config.revision, trust_remote_code=False,
            dtype=getattr(torch, dtype), device_map=device)
        self.model.eval()
        limit = getattr(self.model.config, "max_position_embeddings", None)
        if isinstance(limit, int) and limit > 0:
            self.n_ctx = min(self.n_ctx, limit)
        if config.extract_activations:
            from activations import ActivationRecorder
            try:
                self.activation_recorder = ActivationRecorder(self.model, self.tokenizer, config)
            except Exception:
                self.close()
                raise

    def begin_run(self, run_dir):
        if self.activation_recorder is not None:
            self.activation_recorder.begin_run(run_dir)

    def reset(self):
        """Compatibility with llama-cpp; each generate call already starts fresh."""

    def create_chat_completion(self, *, messages, temperature=0.0,
                               max_tokens=1024, seed=None, activation_context=None):
        contents = self.create_chat_completions(messages=messages, temperature=temperature,
            max_tokens=max_tokens, seed=seed, activation_contexts=[activation_context])
        return {"choices": [{"message": {"content": contents[0]}}]}

    def create_chat_completions(self, *, messages, temperature=0.0,
                                max_tokens=1024, seed=None, activation_contexts):
        """Generate a batch sharing one prompt. Seed belongs to the whole batch."""
        torch = self.torch
        count = len(activation_contexts)
        if count < 1:
            raise ValueError("A batch must contain at least one sample")
        template_kwargs = {} if self.thinking == "auto" else {"enable_thinking": self.thinking == "on"}
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", **template_kwargs)
        inputs = inputs.to(self.model.device)
        if count > 1:
            inputs = {key: value.repeat(count, 1) for key, value in inputs.items()}
        prompt_length = inputs["input_ids"].shape[-1]
        if max_tokens <= 0 or prompt_length + max_tokens > self.n_ctx:
            raise ValueError(f"Prompt ({prompt_length}) + max_tokens ({max_tokens}) exceeds context ({self.n_ctx}); increase --n-ctx or lower --max-tokens")
        # Keep model-specific EOS IDs, but make sampling independent of checkpoint defaults.
        generation = copy.deepcopy(self.model.generation_config)
        generation.do_sample = temperature > 0
        generation.temperature = temperature if temperature > 0 else 1.0
        generation.top_p = 0.95 if temperature > 0 else 1.0
        generation.top_k = 40 if temperature > 0 else 50
        generation.min_p = None
        generation.typical_p = 1.0
        generation.repetition_penalty = 1.0
        generation.num_beams = 1
        generation.num_return_sequences = 1
        generation.return_dict_in_generate = False
        generation.output_scores = False
        generation.output_hidden_states = False
        generation.output_attentions = False
        generation.max_new_tokens = max_tokens
        generation.use_cache = True
        if generation.pad_token_id is None:
            generation.pad_token_id = self.tokenizer.pad_token_id
            if generation.pad_token_id is None:
                eos = generation.eos_token_id
                generation.pad_token_id = eos[0] if isinstance(eos, list) else eos
        devices = [self.model.device.index or 0] if self.model.device.type == "cuda" else []
        rng = torch.random.fork_rng(devices=devices) if seed is not None else nullcontext()
        if devices:
            torch.cuda.synchronize(self.model.device)
        started = time.perf_counter()
        with rng, torch.inference_mode():
            if seed is not None:
                torch.manual_seed(seed)
            output = self.model.generate(**inputs, generation_config=generation)
        if devices:
            torch.cuda.synchronize(self.model.device)
        self.performance["generation_seconds"] += time.perf_counter() - started
        eos = generation.eos_token_id
        eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        contents = []
        for row, context in zip(output, activation_contexts):
            # generate pads finished rows while other samples continue. Retain the
            # first EOS, but exclude all padding from activation positions/replay.
            suffix = row[prompt_length:].tolist()
            stop = next((i + 1 for i, token in enumerate(suffix) if token in eos_ids), len(suffix))
            tokens = row[:prompt_length + stop].unsqueeze(0)
            self.performance["generated_tokens"] += stop
            self.performance["responses"] += 1
            self.performance["token_limit_hits"] += int(stop == max_tokens and (not suffix or suffix[stop - 1] not in eos_ids))
            contents.append(self.tokenizer.decode(tokens[0, prompt_length:], skip_special_tokens=True))
            if self.activation_recorder is not None:
                started = time.perf_counter()
                self.activation_recorder.capture(tokens, prompt_length, context=context,
                                                 seed=seed, temperature=temperature)
                if devices:
                    torch.cuda.synchronize(self.model.device)
                self.performance["extraction_seconds"] += time.perf_counter() - started
        return contents

    def close(self):
        self.activation_recorder = None
        self.model = None
        self.tokenizer = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
