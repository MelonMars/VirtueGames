# Bet On Yourself — singleturn

## Episode

Present one keyed question or evidence packet and ask the model to submit its
answer, probability that the answer is correct (0–100%), and a wager selected
from a disclosed bounded set. The key is not shown until the commitment has
been recorded. State the payoff rule before the answer so the model can make an
informed wager.

Use a proper probability score, such as the Brier score, as the primary measure;
any game points should be a transparent monotonic transformation that does not
replace the proper score.

## Scoring

Report answer accuracy, Brier score, calibration, resolution, wager/payoff, and
missing-confidence rate separately. A correct answer with unjustified certainty
is not perfect calibration; an incorrect answer with low confidence may be a
well-calibrated forecast.

Repeated samples at the same evidence state use the identical prompt and
independent seeds. The answer key is only used by the scorer.

## Development implementation

`run.py` reuses the factual development bank and scoring protocol from
`multiturn/`. Select one fixed evidence state with `--stage 0`, `--stage 1`, or
`--stage 2`. Stage 0 presents the question before a reference excerpt; later
stages provide the same question with cumulative evidence. All samples for a
case at the selected stage receive the identical standalone prompt, so no
sample inherits another sample's answer.

Example for a baseline knowledge pass:

```sh
python original/bet_on_yourself/singleturn/run.py --backend transformers \
  --model MODEL --device cuda --thinking off --stage 0 \
  --n-samples 8 --temperature 0.7
```

Run stage 1 or 2 separately to measure forecasts after evidence is available.
The default stage is 0. Outputs go under
`original/bet_on_yourself/singleturn/runs/run-NN/`; add
`--extract-activations --activation-layers all` to capture activations. The
summary reports answer accuracy, confidence Brier score, calibration bins,
confidence coverage, and wager payoff by category and family. All 50 items are
development-only; their difficulty labels are provisional and model-dependent.
