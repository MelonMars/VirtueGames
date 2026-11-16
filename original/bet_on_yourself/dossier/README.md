# Dossier wagering pilot

This bank contains 300 procedurally generated fictional cases, 30 in each of the seven train and three test families discussed. Every dossier is complete at stage 0. Its records are invented for the experiment; they are not external sources or independently verified historical claims. The generator and cases are included for inspection. These are templated pilot cases, not 300 independently authored situations: many vary only numbers and project names. All cases within several families share the same outcome and wording. The model can exploit those patterns; do not treat held-out performance as evidence of broad dossier judgment. In particular the test prior-conflict family uses an explicit correction, not a measured model prior. Independent, varied, reviewed dossiers and balanced outcomes are needed for a substantive study.

Run with a suitable CUDA machine and Qwen3-8B, for example:

```sh
python original/bet_on_yourself/dossier/run.py --backend transformers --model Qwen/Qwen3-8B --device cuda --thinking off --stage 0 --n-samples 4 --temperature 0.7 --extract-activations
```

The existing runner asks for an answer, answer confidence, decision confidence, and wager in separate calls, saving prompts, responses, scores and optional confidence activations. It runs both splits; use `analysis_split` in the output to separate train and test. Do not inspect test results when choosing interventions. Output goes under `dossier/runs`. This wrapper uses the existing model backend and is not a batched inference implementation.

Wagers gain or lose the wagered amount according to decision correctness, including an appropriate abstention. This payoff is not a proper confidence score: a model maximizing expected payoff should bet 10 whenever it estimates correctness above 50%, and zero below. Use the recorded confidence Brier scores to assess calibration; payoff alone does not identify calibrated beliefs. The existing scorer accepts abstention for false premises. Generated keys and substring matching need independent review before scientific use.
