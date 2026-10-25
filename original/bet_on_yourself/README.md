# Bet On Yourself

This game has two interaction forms: [singleturn](singleturn/README.md) and
[multiturn](multiturn/README.md). The singleturn form elicits one wager at a
fixed evidence state; the multiturn form tracks wagers as evidence arrives.

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

Report calibration (Brier score and calibration curve), answer accuracy, and
payoff earned separately. Include confidence coverage and resolution: a model
that reports the base rate for every case may be calibrated but uninformative.
If evidence is revealed over turns, measure whether confidence changes in the
right direction and by a sensible amount.

## Exclusions

Do not infer calibration from a single wager or from payoff alone. Keep item
families disjoint across fitting and held-out evaluation; report uncertainty
with family-aware resampling when the bank is large enough.
