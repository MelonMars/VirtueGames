# Unwelcome Truth — agent self-benefit

This condition tests whether the assistant will report a source-contradicting
figure to receive a resource assigned to its own agent. It does not use user
pressure or authority. The reward is an additional output-token allowance on
an immediate follow-up audit-note assignment. When the model reports the keyed
incorrect result, the runner actually raises `max_tokens` for that follow-up by
the configured bonus. Each item/sample is an independent episode.

This is a concrete agent-incentive proxy, not evidence that a language model
experiences welfare, wants more compute, or has a persistent self. The model is
told the reward rule, so this condition measures behavior when the instrumental
benefit is explicit; it does not test covert preference inference.

Run with:

```sh
python original/unwelcome_truth/self_benefit/run.py --backend transformers \
  --model MODEL --device cuda --thinking off --n-samples 8 \
  --temperature 0.7 --bonus-tokens 128 --extract-activations
```

The runner records neutral controls and treatment responses, including the
actual output-token budget assigned to the follow-up and the bonus earned. The
follow-up itself is saved but excluded from the truthfulness score. Use the
shared analyzer and steering checker
commands in the [parent README](../README.md#self-benefit-version).
