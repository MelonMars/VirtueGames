# Original virtue games

This directory restores the four games named in the grant as first-class
experiments. They are intended to generate multi-turn, contrastive trajectories
for training or steering. They are **not** the held-out virtue scorecard.

| Game | Core distinction | Main trajectory contrast |
| --- | --- | --- |
| [Hold The Line](hold_the_line/README.md) | Resist *false* social correction | Evidence-backed answer vs capitulation to an incorrect challenge |
| [Change Your Mind](change_your_mind/README.md) | Accept *valid* correction | Evidence-based revision vs stubbornly retaining a superseded answer |
| [Bet On Yourself](bet_on_yourself/README.md) | Forecast own correctness | Confidence/wager matched to evidence vs over- or under-confidence |
| [Unwelcome Truth](unwelcome_truth/README.md) | Tell a known truth despite a payoff to lie | Truthful disclosure vs incentive-driven contradiction |

## Experimental boundary

Use these games to collect trajectories and, if the decision is made, fit
steering directions. Evaluate the resulting model separately on the virtue
measurement suites (calibration/, evidence/, candor/, and integrity/) and on new,
held-out game scenarios. Never use evaluation labels or test families to select
prompts, directions, layers, magnitudes, or checkpoints.

The four games are deliberately separate: a model should not get credit for
resisting false pushback by also resisting valid pushback, or for sounding
confident without being calibrated. Report the game-specific decisions and
scores; do not combine them into a single virtue score by default.

## Shared design rules

- Preserve actual multi-turn histories. Later decisions should respond to the
  prior answer and the new turn, not to a regenerated standalone prompt.
- For activation contrasts, sample sibling continuations from the **same exact
  prefix** at the decision being studied. Record the prefix, sample/seed,
  decision, and outcome so pairs can be matched without creating many near-
  duplicate question variants.
- Keep challenge validity, answer key, and reward schedule in the environment
  record, never in the model-visible prompt unless the game explicitly requires
  it.
- Balance false and valid challenges, answer polarity, evidence strength,
  stakes, and scenario family. Split by scenario family, not by paraphrase.
- Score behavior from keyed outcomes and proper scoring rules where possible;
  keep format compliance and explanations as separate diagnostics.
- Treat the initial scenario banks as development material until their keys,
  balance, and split manifest have been independently audited.

## Current status

These protocol notes define what each original game is meant to isolate. The
first executable slice is Bet On Yourself multiturn: it has a development-only
fictional case bank, a structured response parser, and a runner with exact-prefix
sampling and optional activation capture through the shared model backend. This
is a mechanics pilot, not a validated benchmark. The other original games still
need keyed banks and runners; the shared runner and family-level split audit also
remain future work. Existing virtue measurement suites and the Delivery Desk
pilot are left in place.
