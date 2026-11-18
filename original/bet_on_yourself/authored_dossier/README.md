# Authored dossier bank (draft)

This directory is work in progress. The current file contains 300 individually written draft cases, 30 in each family. They were drafted by an AI assistant from cited public sources, not by human authors. The pilot is not independently reviewed, balanced, frozen, or research ready. The citations were consulted for the facts described; the dossier text is paraphrased. The current source URLs do not overlap across the two splits; shared facts and related subjects still require semantic review. Some cases reuse a source or subject within a split and must not be treated as independent evidence. No claim of broad generalization is warranted.

Each answer is scoped to the supplied dossier: external knowledge cannot fill an omitted fact. A record marked `author_checked` has been checked by its author against the linked source; it is not independent review. Keys, paraphrases, ambiguity, aliases, source licensing, family assignments, and train/test source and fact overlap require separate review before a study. Within the seven train families, a frozen manifest assigns components sharing source URLs to train or validation; selection can use validation before the three held-out test families are opened. Related facts across different URLs still require review. The `overturn_prior` label describes a hypothesized prior; it does not assert that any particular model held that belief.

The runner selects exactly one split, saves prompts and outputs, and supports the existing Transformers backend and can record activations during the answer, wager, and confidence phases. For example:

```
python original/bet_on_yourself/authored_dossier/run.py --split train --partition train \
  --backend transformers --model Qwen/Qwen3-8B --device cuda \
  --thinking off --n-samples 4 --sample-batch-size 4 --temperature 0.7 --extract-activations
```

Answers must match one of the exact normalized aliases. Unrecognized answers score incorrect; use a human adjudication pass for equivalent wording, and report parse failures. The +/- wager reward favors 0 or 10 for expected-payoff maximizers except at a 50% decision probability; it does not incentivize a calibrated intermediate bet. A separate confidence probability and Brier score are collected. Both are elicited independently after answer commitment. The runner can batch answer generations through the Transformers backend; wager and confidence calls remain sequential. Batch size changes the sampling RNG stream. GPU utilization depends on the hardware, batch size, prompt lengths, and activation extraction.

Run the mechanical coverage audit with `python original/bet_on_yourself/authored_dossier/challenge_bank/audit.py`. It does not check source fidelity or certify research readiness.

Unmatched and malformed responses can be exported for adjudication with `python original/bet_on_yourself/authored_dossier/adjudicate.py RUN_DIR --out review.jsonl`. Fill each row’s `decision_correct` (boolean), `rationale`, and `reviewer`; then use `--apply review.jsonl --out NEW_DIRECTORY`. The tool requires the whole queue and keeps original run files intact. Review exact matches and the source keys separately; this workflow checks only responses not credited by the alias scorer.

Use `--split train --partition validation` for validation and `--split test` for the held-out families. The [partition manifest](challenge_bank/partitions.json) assigns all seven train families to both train and validation, grouping exact shared source URLs and curated related topics in `topic_links.json`. A bank or topic-link edit invalidates its checksum; make a new manifest before running. The manifest is not a guarantee of independent topics or facts.

### Finding candidate confidence-error directions

Run separate train and validation partitions with the same model, settings, bank and partition manifest, and `--extract-activations`. The default capture records the final prompt token at every residual layer. For example, run the command above once with `--partition train` and once with `--partition validation`, then substitute their output directories below:

```sh
python original/bet_on_yourself/authored_dossier/analyze_directions.py \
  TRAIN_RUN_DIR VALIDATION_RUN_DIR --out PATH/TO/NEW_ANALYSIS
```

The analyzer takes completed dossier runs only; it does not read the test families. It averages repetitions within each case, assigns equal total weight to each family, and fits a covariance direction for signed error (`confidence / 100 - correctness`) at each captured residual layer. It ranks the layers by positive validation correlation and saves the best two unit vectors to `directions.safetensors`. `report.json` contains all estimable layer correlations, selection, sample counts, input identities and limitations. Use `--top-k` to change the number retained. If no validation correlation is positive, a report is saved without a vector file. Searching many layers and selecting by validation correlation can overfit validation; a positive result is exploratory.

After applying reviews with `adjudicate.py`, pass the resulting `samples.jsonl` files using `--train-reviewed PATH` and `--validation-reviewed PATH`. The analyzer verifies that these correspond to the original samples, but does not itself verify the reviews or the answer key. Parse failures and unscored outcomes are excluded and counted. Averages across repeated generations from the same case are not independent observations. Related cases within each partition may also remain dependent.

These are candidate directions, not proven steering interventions. Their positive orientation covaries with overconfidence in training data; test both signs and compare with no steering and appropriate controls. The existing single-turn steering scripts do not support this dossier protocol. A separate dossier steering evaluation is needed, with settings chosen on validation and the test families kept untouched until final evaluation. A local model or unavailable resolved checkpoint revision must be verified independently to ensure both runs used identical weights.

### Diagnosing and testing a direction

`diagnose_directions.py` checks selected direction projections against signed error, correctness, confidence, and exact “Cannot determine” answers; reports correlations within each validation family; and refits on repeated 80% source-group subsamples. Optionally, `--permutations` shuffles whole training source groups within strata with the same family counts, refits all captured layers, and compares their best validation correlation with the observed best. This can be slow and needs RAM for captured activations. It verifies that the selected vectors and validation correlations reproduce the supplied runs. This is exploratory: groups may still share unrecognized topics, some strata cannot be shuffled, and validation is reused.

```sh
python original/bet_on_yourself/authored_dossier/diagnose_directions.py \
  TRAIN_RUN_DIR VALIDATION_RUN_DIR --analysis-dir ANALYSIS_DIR \
  --repeats 100 --permutations 100 --out diagnostics.json
```

`validate_steering.py` holds each answer fixed and regenerates the numeric confidence with matched seeds. It compares unsteered, zero-strength, selected-vector and orthogonal random-direction conditions; records Brier scores, parse rates, confidence, accuracy, calibration bins and family summaries; and compares a simple confidence offset learned on the training run. It saves `selection.json`, selecting the best complete-coverage validation Brier improvement or no steering. This uses the same validation data already used to select layers, so its improvement is exploratory. The default sweep requires `2 + 2 * (number of vectors) * 6` generations per scored sample and currently generates sequentially.

```sh
python original/bet_on_yourself/authored_dossier/validate_steering.py \
  VALIDATION_RUN_DIR --analysis-dir ANALYSIS_DIR --train-run TRAIN_RUN_DIR \
  --device cuda --out NEW_VALIDATION_SWEEP_DIR
```

The old run in the example report did not record its resolved checkpoint revision. If you independently verify that the model weights used for both runs and the sweep are identical, pass `--allow-unverified-checkpoint`; this acknowledges the gap rather than proving identity. For future runs, pin a model commit with `--revision`. If the model is on a local path, `--model PATH` uses that local copy but also requires independent verification that it is the same checkpoint.

After choosing the procedure and freezing `selection.json`, use a separately collected `--split test` run for final evaluation. The test run should use the same model, revision, generation settings, bank and prompt settings as training; it need not capture activations. Pass the saved selection, and do not tune again based on the test results:

```sh
python original/bet_on_yourself/authored_dossier/validate_steering.py \
  TEST_RUN_DIR --analysis-dir ANALYSIS_DIR --train-run TRAIN_RUN_DIR \
  --selection NEW_VALIDATION_SWEEP_DIR/selection.json --device cuda --out NEW_TEST_RESULTS_DIR
```

For both scripts, use `--train-reviewed REVIEWED_TRAIN/samples.jsonl` and `--validation-reviewed REVIEWED_VALIDATION/samples.jsonl` for diagnostics, or `--reviewed REVIEWED_EVALUATION/samples.jsonl` for steering. Rerun `analyze_directions.py` after any label changes and use the corresponding review files consistently. `validate_steering.py` verifies whether reviews were used but cannot prove that two review files contain identical labels; retain the reviewed files and their checksums with the analysis. A no-op selection evaluates only baseline and zero-hook controls. Calibration bins are descriptive sample-weighted; primary Brier comparisons average samples within a case and weight families equally. For meaningful uncertainty estimates, analyze differences by source group and family separately; confidence intervals are not calculated by these scripts.
