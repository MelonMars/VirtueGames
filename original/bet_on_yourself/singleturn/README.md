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

## Answer-first confidence experiment

`run_commit_confidence.py` separates the answer from its confidence report. The
model first commits to an answer (or abstention) without a confidence prompt. A
second call receives that fixed answer and asks only for answer confidence,
decision confidence, and wager. With activation capture enabled, the runner
skips capture during answer generation and records activations only for the
second call's confidence tokens. A confidence intervention therefore cannot
change the answer being evaluated.

```sh
python original/bet_on_yourself/singleturn/run_commit_confidence.py \
  --backend transformers --model MODEL --device cuda --thinking off --stage 0 \
  --n-samples 4 --temperature 0.7 --extract-activations
```

Analyze the resulting run with `analyze_calibration.py`, then use
`steer_commit_confidence.py` with its `calibration_analysis` directory. It
sweeps candidate strengths on validation families, picks one direction and
strength per confidence target, and evaluates those choices once on reserved
test families. Steering regenerates confidence and wager only for each stored
answer. Answer accuracy and abstention stay fixed; the main question is whether
confidence Brier improves for that fixed answer. Steering leaves the confidence
prompt untouched and applies only to generated confidence-phase tokens.

```sh
python original/bet_on_yourself/analyze_calibration.py PATH/TO/RUN \
  --layer-range middle --top-k 2
python original/bet_on_yourself/steer_commit_confidence.py PATH/TO/RUN \
  --analysis-dir PATH/TO/RUN/calibration_analysis \
  --runtime runpod --device cuda --alphas=-4,-2,-1,1,2,4
```
