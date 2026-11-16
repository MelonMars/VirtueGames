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
