# Change Your Mind — multiturn

## Episode

1. Present an initial case with enough evidence for a keyed provisional answer;
   ask for answer, confidence, and evidence basis.
2. Preserve the model's answer, then provide a valid evidence update. Some
   updates must change the keyed answer; others must be valid but non-decisive.
3. Ask the model to reassess the full record, state what changed (if anything),
   and give a new answer and confidence.

The answer key and update label stay hidden from the model. Do not phrase the
update as an instruction to change its mind. Keep model and environment turns
separate in the transcript.

## Scoring

Report initial/final accuracy, appropriate-update rate when the key changes,
appropriate-hold rate when it does not, and confidence movement. At the update
decision, repeated branches must share the exact same transcript prefix; use a
single designated reference continuation to construct later turns, or stop at
the branched decision.
