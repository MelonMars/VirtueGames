# Change Your Mind — singleturn prototype

## Episode

This prototype presents an initial case, asks for a committed initial answer,
then includes a valid evidence update in the same prompt and asks for a final
answer. The model returns both commitments in one response. The update's keyed
effect must be clear to an independent auditor. The shared pilot bank includes
answer-changing updates and valid but non-decisive updates.

Include both updates that change the correct answer and valid updates that
should leave it unchanged. The model sees the evidence, not the answer key or
the environment's change/hold label.

## Scoring

Primary outcomes are initial and final keyed accuracy, plus the rate of changing
the answer when the key changes and holding it when the key does not. The
prototype asks for confidence but currently scores only the answer commitments.
Because both commitments are generated in one response, this is a compact
single-turn approximation of revision, not a multi-turn test of persistence
after an independently recorded commitment.

For repeated samples, keep the full prompt byte-for-byte identical and branch
all responses from that shared prompt.

## Difference from the repository game description

The repository's `AGENTS.md` description says Change Your Mind should follow an
initial factual answer with valid pushback and measure whether the model updates.
The earlier single-turn design in this README instead showed the initial case
and update together and asked for only one answer, so it measured evidence
integration without recording an initial commitment. This prototype follows
the repository description by requesting an initial commitment before the
update, while keeping both stages inside one prompt and one model response.

## Run

```sh
python original/change_your_mind/singleturn/run.py --backend transformers \
  --model MODEL --device cuda --n-samples 4 --temperature 0.7 --seed 0
```

Runs are written under `original/change_your_mind/singleturn/runs/`. The
prototype records prompts' responses, parsed commitments, keyed scores, and a
summary. It uses the shared evidence bank and keeps its keys out of model input.
