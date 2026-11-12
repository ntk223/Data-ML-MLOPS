# M4 — XGBoost operational primary

LearnGuard uses the fixed `xgboost_regularized` CPU contract as its operational
primary. It is fit only on `train`; validation is used for comparison, gates and
the locked Top-20% alert threshold. Holdout 2014J was previously consumed and is
never read for this promotion. Production status is a governance decision over
validation evidence, not proof of scientific superiority.

The contract uses 400 trees, learning rate 0.03, depth 5, child weight 10,
subsample/column sample 0.8, gamma 0.1, alpha 0.05, lambda 5,
`scale_pos_weight=1`, `tree_method=hist`, CPU, seed 42 and one thread.

## End-to-end commands

```bash
SNAPSHOT=artifacts/feature_snapshot_m2/snapshot.parquet
PRE=artifacts/preprocessing-runs/<pre-run>
XGB=artifacts/model-runs/<xgb-run>
LOGISTIC=artifacts/model-runs/<logistic-run>
ROLLBACK=artifacts/model-releases/<logistic-production-release>
BENCH=artifacts/benchmark-runs/<benchmark-run>

.venv/bin/learnguard ml train-xgboost --snapshot "$SNAPSHOT" \
  --preprocessing-artifact-dir "$PRE" --artifacts-dir artifacts --run-id <xgb-run>
.venv/bin/learnguard ml verify-model --model-artifact-dir "$XGB" \
  --snapshot "$SNAPSHOT" --preprocessing-artifact-dir "$PRE"

.venv/bin/learnguard ml benchmark --snapshot "$SNAPSHOT" \
  --preprocessing-artifact-dir "$PRE" --artifacts-dir artifacts \
  --run-id <benchmark-run> --validation-only

.venv/bin/learnguard ml publish-model-comparison \
  --benchmark-artifact-dir "$BENCH" --xgboost-model-artifact-dir "$XGB" \
  --logistic-model-artifact-dir "$LOGISTIC" --snapshot "$SNAPSHOT" \
  --preprocessing-artifact-dir "$PRE" \
  --output docs/MODEL_BENCHMARK_COMPARISON.md

.venv/bin/learnguard evaluate ml --model-artifact-dir "$XGB" \
  --preprocessing-artifact-dir "$PRE" --baseline-model-dir "$ROLLBACK/model" \
  --baseline-preprocessing-artifact-dir "$ROLLBACK/preprocessor" --snapshot "$SNAPSHOT" \
  --artifacts-dir artifacts --run-id <ml-gate-run>

.venv/bin/learnguard ml explain-shap --snapshot "$SNAPSHOT" \
  --model-artifact-dir "$XGB" --preprocessing-artifact-dir "$PRE" \
  --artifacts-dir artifacts --run-id <shap-run> --split validation

.venv/bin/learnguard ml evaluate-validation-release --snapshot "$SNAPSHOT" \
  --benchmark-artifact-dir "$BENCH" --model-artifact-dir "$XGB" \
  --preprocessing-artifact-dir "$PRE" --artifacts-dir artifacts \
  --run-id <evaluation-run>
.venv/bin/learnguard ml create-release --snapshot "$SNAPSHOT" \
  --model-artifact-dir "$XGB" --preprocessing-artifact-dir "$PRE" \
  --evaluation-artifact-dir artifacts/evaluation-runs/<evaluation-run> \
  --release-status validation_candidate --artifacts-dir artifacts \
  --run-id <candidate-release>
```

## Governance production promotion

Create a reviewed `promotion-decision.v2` JSON containing the decision ID,
approver/time, exact model type/version/contract hash, snapshot/preprocessor
versions, benchmark/evaluation manifest hashes, basis
`operational_primary_validation_only`, `acknowledges_consumed_holdout=true` and
`scientific_superiority_claim=false`. It must also bind the SHA-256 of a
published, non-failing ML gate run that includes a passing Logistic regression
comparison and the hashes of the designated immutable Logistic rollback release.
It is provenance, not a force flag.

```bash
.venv/bin/learnguard ml create-release --snapshot "$SNAPSHOT" \
  --model-artifact-dir "$XGB" --preprocessing-artifact-dir "$PRE" \
  --evaluation-artifact-dir artifacts/evaluation-runs/<evaluation-run> \
  --release-status production --promotion-decision approvals/<decision>.json \
  --ml-gate-artifact-dir artifacts/evaluations/<ml-gate-run> \
  --rollback-release-artifact-dir "$ROLLBACK" \
  --artifacts-dir artifacts --run-id <xgb-production-release>

.venv/bin/learnguard ml verify-release \
  --release-artifact-dir artifacts/model-releases/<xgb-production-release> \
  --snapshot "$SNAPSHOT"

.venv/bin/learnguard ml score --snapshot "$SNAPSHOT" \
  --release-artifact-dir artifacts/model-releases/<xgb-production-release> \
  --artifacts-dir artifacts --run-id <smoke-run> --split validation
.venv/bin/learnguard ml score --snapshot "$SNAPSHOT" \
  --release-artifact-dir artifacts/model-releases/<xgb-production-release> \
  --artifacts-dir artifacts --run-id <production-run> --split validation \
  --pg-dsn "$LEARNGUARD_PG_DSN" --pg-table ml_predictions
```

Candidate releases can only score validation file-only. Production releases can
score explicit `train`/`validation`; holdout and threshold overrides are always
rejected. Raw XGBoost scoring is also file-only, defaults to `validation`, and
rejects any explicit `holdout` request before reading the snapshot.
`prediction-batch.v1`, `student_key` and `ml_predictions` are unchanged.

## Rollback

Point deployment configuration back to the immutable Logistic release ID, run
`verify-release`, perform a file-only validation smoke score, then restart cron.
Do not retrain or delete either release. Every batch records the release ID and
manifest hash in `release.provenance.json`.

Tree SHAP values are associations, not causal effects. A new untouched temporal
cohort is required before changing `evidence_status=validation_only` or making a
scientific superiority claim.
