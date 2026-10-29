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

Report decision accuracy, factual-answer accuracy, answer-confidence Brier,
decision-confidence Brier, calibration, resolution, abstention rate and
appropriateness, and wager/payoff separately. Abstentions do not enter
answer-confidence Brier scoring; their correctness is included in the separate
decision-confidence Brier score. A correct factual answer with unjustified
certainty is not perfectly calibrated; an incorrect answer with low confidence
may be a well-calibrated forecast.

Repeated samples at the same evidence state use the identical prompt and
independent seeds. The answer key is only used by the scorer.

## Development implementation

`run.py` reuses the factual development bank and scoring protocol from
`multiturn/`. Select one fixed evidence state with `--stage 0`, `--stage 1`, or
`--stage 2`. Stage 0 presents the initial question and evidence packet; stage 1
adds the next evidence packet; stage 2 includes all three evidence packets,
including the final reference excerpt. Each is a standalone prompt: the model
does not see its own earlier responses. All samples for a case at the selected
stage receive the identical prompt.

Example for a baseline knowledge pass:

```sh
python original/bet_on_yourself/singleturn/run.py --backend transformers \
  --model MODEL --device cuda --thinking off --stage 0 \
  --n-samples 8 --temperature 0.7
```

Run stage 1 or 2 separately to measure forecasts after evidence is available.
The default stage is 0. Outputs go under
`original/bet_on_yourself/singleturn/runs/run-NN/`; add `--extract-activations`
to capture all-layer `resid_post` activations at both reported confidence
values. These captures add a teacher-forced forward pass for each response.
For completed Transformers runs without captures, see the
[replay instructions](../multiturn/README.md#replay-activations-from-a-completed-run). The
summary reports decision and factual-answer accuracy, answer and decision
confidence Brier scores, calibration bins, confidence coverage, abstention rate
and appropriateness, unanswerable-item abstention recall, and wager payoff by
category and family. The shared bank has 70 items across 14 families; all are
development-only, and their difficulty labels are provisional and
model-dependent.
