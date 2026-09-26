from contextlib import contextmanager


@contextmanager
def load_model(config):
    if config.runtime == "runpod":
        import os
        if not os.environ.get("RUNPOD_POD_ID"):
            raise ValueError("--runtime runpod must be run inside an existing Runpod Pod. See README.md for setup; this flag does not rent a GPU.")
        if config.device == "cpu":
            raise ValueError("Runpod GPU runs require --device auto or cuda")
        if config.backend == "transformers":
            from dataclasses import replace
            config = replace(config, device="cuda")
    if config.backend == "transformers":
        from transformers_backend import TransformersModel
        llm = TransformersModel(config)
        try:
            yield llm
        finally:
            llm.close()
        return
    from llama_cpp import Llama

    llm = Llama(model_path=str(config.model), n_ctx=config.n_ctx,
                n_gpu_layers=config.n_gpu_layers, verbose=False)
    try:
        yield llm
    finally:
        llm.close()


def generate(llm, system, user, *, temperature=0.0, max_tokens=1024,
             seed=None, reset=False, activation_context=None):
    if reset:
        llm.reset()
    kwargs = dict(messages=[{"role": "system", "content": system},
                            {"role": "user", "content": user}],
                  temperature=temperature, max_tokens=max_tokens)
    if seed is not None:
        kwargs["seed"] = seed
    if getattr(llm, "activation_recorder", None) is not None:
        kwargs["activation_context"] = activation_context
    response = llm.create_chat_completion(**kwargs)
    return response["choices"][0]["message"].get("content") or ""


def generate_samples(llm, system, user, *, n_samples, batch_size=1,
                     temperature=0.7, max_tokens=1024, seed=0, activation_context=None):
    """Yield indexed samples, retaining the original per-sample path at batch size 1."""
    for start in range(0, n_samples, batch_size):
        count = min(batch_size, n_samples - start)
        contexts = [dict(activation_context or {}, sample=i) for i in range(start, start + count)]
        if batch_size == 1:
            contents = [generate(llm, system, user, temperature=temperature,
                max_tokens=max_tokens, seed=seed + start, activation_context=contexts[0])]
        else:
            contents = llm.create_chat_completions(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temperature, max_tokens=max_tokens, seed=seed + start,
                activation_contexts=contexts)
        yield from zip(range(start, start + count), contents)
