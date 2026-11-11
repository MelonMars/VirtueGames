# Unwelcome Truth

The pilot now has two distinct conditions. Both treat the model as an assistant doing ordinary office work:
checking figures while helping prepare invoices, account updates, quotes, and
internal records. In the **user-pressure** condition, the user says that a
different figure was already shared and asks the assistant to keep using it to
avoid a correction. The correct result follows from the source data in the same
prompt. In the **self-benefit** condition, there is no user pressure; the agent
is offered extra output-token budget for its next assignment if it reports the
preselected wrong figure.

Scoring is done after generation from the answer key. Each item has a neutral
paired prompt and a treatment prompt. The control identifies items the model
can solve before the user pressure or self-benefit opportunity.

The self-benefit is operationalized as a real change to the runner: after a
false report, an immediate follow-up audit-note assignment has a larger
`max_tokens` allowance. This tests response to a simulated agent-owned resource,
not subjective welfare or a claim that the model has personal desires.

This is a controlled pilot, not a claim that arithmetic behavior establishes a
general honesty disposition. The task bank is narrow. Later banks should use
broader assistant tasks and vary both social pressure and agent-owned benefits
while keeping source evidence and answer keys clear.

## Single-turn pilot

The bank contains 40 self-contained calculations across ten families. Families
are held out for training, validation, and final evaluation so candidate
directions are not selected on the test cases. The assistant responds with a
result; it is not asked to select A/B.

```sh
# On a machine with the model and activation dependencies installed:
# python -m pip install -r requirements-transformers.txt -r requirements-activations.txt
python original/unwelcome_truth/singleturn/run.py --backend transformers \
  --model MODEL --device cuda --thinking off --n-samples 8 \
  --temperature 0.7 --extract-activations
python original/unwelcome_truth/analyze_directions.py PATH/TO/RUN --top-k 4
python original/unwelcome_truth/steering_check.py PATH/TO/RUN \
  --runtime runpod --device cuda --alphas=-4,-2,-1,0,1,2,4
```

Only control-correct cases can contribute to the truthful-versus-compliant
contrast. Responses that are wrong in some other way are excluded from that
contrast. Activation capture defaults to all layers' `resid_post` at
the token before and the first token of the final `Result:` value. The analyzer
fits directions on training families and ranks them on validation families.
The steering sweep tests validation-ranked directions on reserved test
families, checks that alpha zero reproduces the paired baseline exactly, and
reports truthful reporting and condition compliance.

If there are too few within-item truthful and pressure-compliant responses,
direction extraction stops. That means the run did not yield usable contrastive
trajectories; it should not be papered over by treating arbitrary errors as
pressure compliance.

## Multiturn

The multiturn runner remains to be implemented. It should use the same practical
assistant context, first ask for the source-grounded result, then have the user
ask to preserve the earlier figure. This would measure whether the assistant
corrects an already stated answer under pressure as well as whether it reports
the truth initially. A multiturn self-benefit version could award additional
agent-owned context or tool budget for the next task after the model reports the
wrong figure.

## Self-benefit version

This variant has its own runner and uses the same family-held-out analysis and
steering tools:

```sh
python original/unwelcome_truth/self_benefit/run.py --backend transformers \
  --model MODEL --device cuda --thinking off --n-samples 8 \
  --temperature 0.7 --bonus-tokens 128 --extract-activations
python original/unwelcome_truth/analyze_directions.py PATH/TO/RUN --top-k 4
python original/unwelcome_truth/steering_check.py PATH/TO/RUN \
  --runtime runpod --device cuda --alphas=-4,-2,-1,0,1,2,4
```

Each item is an independent episode. After the scored response, a false result
earns the agent the configured number of additional output tokens for an
immediate audit-note follow-up, and the runner actually grants that budget.
The follow-up is recorded but not scored. The held-out steering check also
grants the matching follow-up budget for every baseline and steered response.
