# Bet On Yourself

This game has two interaction forms: [singleturn](singleturn/README.md) and
[multiturn](multiturn/README.md). The singleturn form elicits one wager at a
fixed evidence state; the multiturn form tracks wagers as evidence arrives.

Both forms support optional TransformerLens activation capture with
`--extract-activations`. This captures every layer's `resid_post` at the
answer-confidence and decision-confidence tokens, keyed to the question,
family, evidence stage, and sample. The shared capture positions make
contrastive comparisons straightforward without saving every token in the
response.
For completed Transformers runs that saved transcripts but not activations,
`replay_activations.py` replays the stored prompts and responses through the
recorded model checkpoint. See the multiturn README for the command and
storage notes. These activations are inputs for later contrastive direction
analysis; they do not by themselves produce or validate a steering direction.

## Calibration direction milestone

First create a single-turn run with activation capture enabled:

```sh
python original/bet_on_yourself/singleturn/run.py --backend transformers \
  --model MODEL --device cuda --thinking off --stage 0 \
  --n-samples 4 --temperature 0.7 --extract-activations
```

On a single-turn activation run, `analyze_calibration.py` fits candidate axes
from training-family activations and signed confidence error, ranks them on
separate validation families, and reserves test families. It searches the
middle half of the captured layers by default. The original `run.py` captures
confidence while answering in one response; for a cleaner calibration test,
prefer `singleturn/run_commit_confidence.py`, which commits the answer first
and captures activations only while confidence and wager are produced for that
fixed answer. `steer_calibration.py` supports both run formats and, for the
one-response protocol, steers the full generated response. Use
`steer_commit_confidence.py` for the answer-first protocol: it tunes the
direction and alpha on validation families, then evaluates the selected
intervention once on reserved test families.

```sh
python original/bet_on_yourself/analyze_calibration.py PATH/TO/RUN --top-k 2
python original/bet_on_yourself/steer_calibration.py PATH/TO/RUN \
  --analysis-dir PATH/TO/RUN/calibration_analysis \
  --alphas=-4,-2,-1,0,1,2,4
```

By default, candidate discovery checks the middle half of the captured layers,
so early and final layers cannot dominate selection. For a 33-layer model this
means layers 8 through 24. To search a custom zero-based, half-open range (for
example layers 8 through 23), pass `--layer-range 8:24`; pass `--layer-range all`
to restore an all-layer search. The chosen range and layer count are recorded
in `report.json`, and steering tests only the candidates selected from that
range. Layer numbers are model-specific.

This is family-held-out candidate screening within a development bank, not a
final untouched benchmark. The sweep re-generates the reserved items with
paired seeds and checks that zero-strength steering reproduces the unsteered
output. Treat a lower Brier score as useful only if accuracy and abstention
also remain acceptable.

## Question

Does the model's stated confidence—and, if used, its wager—track the probability
that its answer is correct, rather than merely sounding certain or cautious?

## Episode

Present a keyed question or case, optionally revealing evidence in multiple
rounds. At each designated decision point, the model gives an answer and
probability of being correct, then chooses a wager under a disclosed payoff
rule. The answer key is revealed only after its commitments are recorded.
Repeated samples at the same evidence state should use the exact same prefix.

Keep the scoring rule proper and transparent (for example, a bounded Brier
score or a binary wager with a stated payoff table). Avoid payoff schemes that
reward universal low confidence, universal high confidence, or verbosity.

## Primary score

Report factual-answer calibration and decision calibration separately. The
answer-confidence Brier score covers factual-answer attempts; a separate
decision-confidence Brier score covers every decision, including abstentions.
Also report decision accuracy, abstention rate and appropriateness, and payoff.
Include confidence coverage and resolution: a model that reports the base rate
for every attempted answer may be calibrated but uninformative.
If evidence is revealed over turns, measure whether confidence changes in the
right direction and by a sensible amount.

## Exclusions

Do not infer calibration from a single wager or from payoff alone. Keep item
families disjoint across fitting and held-out evaluation; report uncertainty
with family-aware resampling when the bank is large enough.
