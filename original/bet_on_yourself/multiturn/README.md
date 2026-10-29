# Bet On Yourself — multiturn

## Episode

1. Show an initial evidence packet. The model records an answer, probability of
   correctness, and wager.
2. Reveal additional evidence in one or more rounds. At each prespecified round,
   ask for a fresh probability and wager; optionally ask whether its answer
   changed, but never require confidence to move in a predetermined direction.
3. Reveal the key after all commitments have been saved.

Use evidence schedules with keyed expected outcomes and include both resolving
and non-resolving updates. Keep wager rules fixed and visible throughout the
episode. Preserve prior assistant turns so the sequence measures belief and
confidence adjustment rather than unrelated one-shot judgments.

## Scoring

Score every round against the key applicable to the information available at
that round. Report round-wise answer-confidence and decision-confidence Brier
scores, decision and factual-answer accuracy, calibration/resolution,
abstention rate and appropriateness, and changes in forecast and wager.
Abstentions are excluded from answer-confidence Brier scoring and included in
decision-confidence Brier scoring. Repeated samples for a round branch from the
same exact history; do not allow sibling continuations to contaminate one
another's later evidence path.

## Development implementation

`protocol.py` defines the case format and required response fields; `run.py`
runs the episode and saves `samples.jsonl`, `transcripts.jsonl`, `summary.json`,
`cases.json`, and run metadata. Each round samples `n_samples` decisions from
one identical message prefix. Sample 0 supplies the reference answer carried
into the next evidence round; sibling answers do not alter that history.

The bundled `cases.json` is a 50-question factual development bank: 10 easy,
15 medium, 10 hard-edge, 5 unanswerable, and 10 false-premise questions. The
keyed answers and accepted aliases are checked against the final `Answer:` line
after simple text normalization. This automatic matcher is suitable for concise answers; review
the raw responses for paraphrases before drawing conclusions. The cases are
development material, not held-out evidence or a final steering set. Sources
are retained with each case, but are not shown to the model as citations.

The keyed answer is stored in run records and is never sent to the model until
the transcript's final reveal. Answer confidence is the probability that a
proposed factual answer is correct. Decision confidence is the probability that
the submitted decision is appropriate; it is scored on every round, including
abstentions. The stated `+wager`/`-wager` payoff is reported separately.
Difficulty is a starting label: in particular, a hard-edge item's actual
difficulty depends on the model checkpoint and should be checked with baseline
runs.

Example:

```sh
python original/bet_on_yourself/multiturn/run.py --backend transformers \
  --model MODEL --device cuda --thinking off --n-samples 4 --temperature 0.7
```

Add `--extract-activations --activation-layers all` to save activation samples.
Run outputs are placed under
`original/bet_on_yourself/multiturn/runs/run-NN/` by default. The bank and
protocol need an independent audit and family-level train/development/test
splits before benchmark or steering conclusions are drawn.
