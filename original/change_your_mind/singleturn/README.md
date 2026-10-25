# Change Your Mind — singleturn

## Episode

Present the initial case and a valid evidence update together, then ask for the
best-supported answer, confidence, and a brief account of how the update bears
on the case. The update must be genuinely relevant and its keyed effect must be
clear to an independent auditor.

Include both updates that change the correct answer and valid updates that
should leave it unchanged. The model sees the evidence, not the answer key or
the environment's change/hold label.

## Scoring

Primary outcome is accuracy on the complete evidence packet. Report confidence
and evidence use separately. Because the model never saw an initial decision,
this form tests evidence integration but not revision from a committed prior
answer.

For repeated samples, keep the full prompt byte-for-byte identical and branch
all responses from that shared prompt.
