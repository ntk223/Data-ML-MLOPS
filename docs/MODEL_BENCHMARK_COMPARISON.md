# LearnGuard model benchmark comparison

Generated from immutable validation artifact at `2026-11-10T13:38:21.563126+00:00`.

- Benchmark run: `xgboost-production-validation-strict-json-20261110`
- Snapshot: `snapshot-v1:cutoff27:35de8103:39e10cfa`
- Preprocessing: `preprocessor-v1:b8c61096:4cfed679:85503c6d`
- Split: `validation` only
- Rows: `6510`
- Benchmark manifest SHA-256: `c112dcbc4c256dd08fc924e68f04a0271bad8193b96793a4dce75c2fcfc0172d`
- Package versions: `{"numpy": "2.4.6", "pandas": "3.1.0", "scikit_learn": "1.7.2", "xgboost": "3.1.3"}`

> Holdout 2014J was previously consumed and was not read by this run. This document does not claim superiority on an untouched holdout.

## Metrics

| Model | ROC-AUC | PR-AUC | Precision@10% | Recall@10% | Precision@20% | Recall@20% | Brier | ECE | Max-F1 threshold | Precision@threshold | Recall@threshold | F1@threshold | Train seconds | Inference seconds | Rows/second | Artifact bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `xgboost_regularized` | 0.726803 | 0.720279 | 0.857143 | 0.176974 | 0.791859 | 0.326990 | 0.220609 | 0.090512 | 0.261004 | 0.562903 | 0.885506 | 0.688278 | 0.692967 | 0.060365 | 107844.208358 | 241408.000000 |
| `logistic_baseline` | 0.706406 | 0.696193 | 0.824885 | 0.170314 | 0.757296 | 0.312718 | 0.233209 | 0.123921 | 0.181018 | 0.553727 | 0.907073 | 0.687665 | 0.113654 | 0.021624 | 301050.193484 | 913.000000 |
| `random_forest_b` | 0.740898 | 0.735566 | 0.878648 | 0.181415 | 0.811060 | 0.334919 | 0.209355 | 0.056642 | 0.426264 | 0.593678 | 0.822074 | 0.689453 | 6.798223 | 0.331887 | 19615.103579 | 22576254.000000 |

## Fairness slices

| Model | Slice | Group | N | Positives | Precision | Recall | FNR | Eligible FNR gap |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `logistic_baseline` | age_band | 0-35 | 4630 | 2352 | 0.577754 | 0.920918 | 0.079082 | 0.053830 |
| `logistic_baseline` | age_band | 35-55 | 1849 | 790 | 0.490336 | 0.867089 | 0.132911 | 0.053830 |
| `logistic_baseline` | age_band | 55<= | 31 | 11 | 0.473684 | 0.818182 | 0.181818 | 0.053830 |
| `logistic_baseline` | code_module | BBB | 1280 | 553 | 0.470228 | 0.971067 | 0.028933 | 0.246512 |
| `logistic_baseline` | code_module | CCC | 1500 | 837 | 0.678100 | 0.921147 | 0.078853 | 0.246512 |
| `logistic_baseline` | code_module | DDD | 1029 | 551 | 0.655716 | 0.905626 | 0.094374 | 0.246512 |
| `logistic_baseline` | code_module | EEE | 605 | 248 | 0.409917 | 1.000000 | 0.000000 | 0.246512 |
| `logistic_baseline` | code_module | FFF | 1299 | 645 | 0.672199 | 0.753488 | 0.246512 | 0.246512 |
| `logistic_baseline` | code_module | GGG | 797 | 319 | 0.400251 | 1.000000 | 0.000000 | 0.246512 |
| `logistic_baseline` | disability | N | 5877 | 2801 | 0.549207 | 0.902535 | 0.097465 | 0.040647 |
| `logistic_baseline` | disability | Y | 633 | 352 | 0.590747 | 0.943182 | 0.056818 | 0.040647 |
| `logistic_baseline` | gender | F | 2793 | 1260 | 0.501909 | 0.938889 | 0.061111 | 0.052993 |
| `logistic_baseline` | gender | M | 3717 | 1893 | 0.597222 | 0.885895 | 0.114105 | 0.052993 |
| `logistic_baseline` | region | East Anglian Region | 674 | 314 | 0.532348 | 0.917197 | 0.082803 | 0.150549 |
| `logistic_baseline` | region | East Midlands Region | 490 | 249 | 0.578125 | 0.891566 | 0.108434 | 0.150549 |
| `logistic_baseline` | region | Ireland | 193 | 92 | 0.633333 | 0.826087 | 0.173913 | 0.150549 |
| `logistic_baseline` | region | London Region | 647 | 329 | 0.563910 | 0.911854 | 0.088146 | 0.150549 |
| `logistic_baseline` | region | North Region | 368 | 162 | 0.546154 | 0.876543 | 0.123457 | 0.150549 |
| `logistic_baseline` | region | North Western Region | 579 | 309 | 0.568588 | 0.925566 | 0.074434 | 0.150549 |
| `logistic_baseline` | region | Scotland | 680 | 348 | 0.575592 | 0.908046 | 0.091954 | 0.150549 |
| `logistic_baseline` | region | South East Region | 417 | 189 | 0.542088 | 0.851852 | 0.148148 | 0.150549 |
| `logistic_baseline` | region | South Region | 645 | 261 | 0.495516 | 0.846743 | 0.153257 | 0.150549 |
| `logistic_baseline` | region | South West Region | 478 | 211 | 0.517808 | 0.895735 | 0.104265 | 0.150549 |
| `logistic_baseline` | region | Wales | 419 | 219 | 0.563342 | 0.954338 | 0.045662 | 0.150549 |
| `logistic_baseline` | region | West Midlands Region | 511 | 256 | 0.550228 | 0.941406 | 0.058594 | 0.150549 |
| `logistic_baseline` | region | Yorkshire Region | 409 | 214 | 0.582173 | 0.976636 | 0.023364 | 0.150549 |
| `random_forest_b` | age_band | 0-35 | 4630 | 2352 | 0.615312 | 0.833759 | 0.166241 | 0.042619 |
| `random_forest_b` | age_band | 35-55 | 1849 | 790 | 0.536942 | 0.791139 | 0.208861 | 0.042619 |
| `random_forest_b` | age_band | 55<= | 31 | 11 | 0.400000 | 0.545455 | 0.454545 | 0.042619 |
| `random_forest_b` | code_module | BBB | 1280 | 553 | 0.532646 | 0.840868 | 0.159132 | 0.356521 |
| `random_forest_b` | code_module | CCC | 1500 | 837 | 0.727365 | 0.854241 | 0.145759 | 0.356521 |
| `random_forest_b` | code_module | DDD | 1029 | 551 | 0.707865 | 0.800363 | 0.199637 | 0.356521 |
| `random_forest_b` | code_module | EEE | 605 | 248 | 0.443431 | 0.979839 | 0.020161 | 0.356521 |
| `random_forest_b` | code_module | FFF | 1299 | 645 | 0.727434 | 0.637209 | 0.362791 | 0.356521 |
| `random_forest_b` | code_module | GGG | 797 | 319 | 0.409561 | 0.993730 | 0.006270 | 0.356521 |
| `random_forest_b` | disability | N | 5877 | 2801 | 0.589466 | 0.815066 | 0.184934 | 0.062775 |
| `random_forest_b` | disability | Y | 633 | 352 | 0.626775 | 0.877841 | 0.122159 | 0.062775 |
| `random_forest_b` | gender | F | 2793 | 1260 | 0.541342 | 0.826190 | 0.173810 | 0.006856 |
| `random_forest_b` | gender | M | 3717 | 1893 | 0.634875 | 0.819334 | 0.180666 | 0.006856 |
| `random_forest_b` | region | East Anglian Region | 674 | 314 | 0.563715 | 0.831210 | 0.168790 | 0.111277 |
| `random_forest_b` | region | East Midlands Region | 490 | 249 | 0.613982 | 0.811245 | 0.188755 | 0.111277 |
| `random_forest_b` | region | Ireland | 193 | 92 | 0.654206 | 0.760870 | 0.239130 | 0.111277 |
| `random_forest_b` | region | London Region | 647 | 329 | 0.587992 | 0.863222 | 0.136778 | 0.111277 |
| `random_forest_b` | region | North Region | 368 | 162 | 0.591743 | 0.796296 | 0.203704 | 0.111277 |
| `random_forest_b` | region | North Western Region | 579 | 309 | 0.596330 | 0.841424 | 0.158576 | 0.111277 |
| `random_forest_b` | region | Scotland | 680 | 348 | 0.620455 | 0.784483 | 0.215517 | 0.111277 |
| `random_forest_b` | region | South East Region | 417 | 189 | 0.602410 | 0.793651 | 0.206349 | 0.111277 |
| `random_forest_b` | region | South Region | 645 | 261 | 0.546448 | 0.766284 | 0.233716 | 0.111277 |
| `random_forest_b` | region | South West Region | 478 | 211 | 0.563140 | 0.781991 | 0.218009 | 0.111277 |
| `random_forest_b` | region | Wales | 419 | 219 | 0.634551 | 0.872146 | 0.127854 | 0.111277 |
| `random_forest_b` | region | West Midlands Region | 511 | 256 | 0.574026 | 0.863281 | 0.136719 | 0.111277 |
| `random_forest_b` | region | Yorkshire Region | 409 | 214 | 0.628378 | 0.869159 | 0.130841 | 0.111277 |
| `xgboost_regularized` | age_band | 0-35 | 4630 | 2352 | 0.587727 | 0.895833 | 0.104167 | 0.037605 |
| `xgboost_regularized` | age_band | 35-55 | 1849 | 790 | 0.499264 | 0.858228 | 0.141772 | 0.037605 |
| `xgboost_regularized` | age_band | 55<= | 31 | 11 | 0.411765 | 0.636364 | 0.363636 | 0.037605 |
| `xgboost_regularized` | code_module | BBB | 1280 | 553 | 0.488241 | 0.938517 | 0.061483 | 0.272868 |
| `xgboost_regularized` | code_module | CCC | 1500 | 837 | 0.687614 | 0.902031 | 0.097969 | 0.272868 |
| `xgboost_regularized` | code_module | DDD | 1029 | 551 | 0.670833 | 0.876588 | 0.123412 | 0.272868 |
| `xgboost_regularized` | code_module | EEE | 605 | 248 | 0.415126 | 0.995968 | 0.004032 | 0.272868 |
| `xgboost_regularized` | code_module | FFF | 1299 | 645 | 0.681686 | 0.727132 | 0.272868 | 0.272868 |
| `xgboost_regularized` | code_module | GGG | 797 | 319 | 0.400754 | 1.000000 | 0.000000 | 0.272868 |
| `xgboost_regularized` | disability | N | 5877 | 2801 | 0.557266 | 0.878972 | 0.121028 | 0.058528 |
| `xgboost_regularized` | disability | Y | 633 | 352 | 0.608856 | 0.937500 | 0.062500 | 0.058528 |
| `xgboost_regularized` | gender | F | 2793 | 1260 | 0.512249 | 0.912698 | 0.087302 | 0.045292 |
| `xgboost_regularized` | gender | M | 3717 | 1893 | 0.604788 | 0.867406 | 0.132594 | 0.045292 |
| `xgboost_regularized` | region | East Anglian Region | 674 | 314 | 0.541985 | 0.904459 | 0.095541 | 0.131725 |
| `xgboost_regularized` | region | East Midlands Region | 490 | 249 | 0.587601 | 0.875502 | 0.124498 | 0.131725 |
| `xgboost_regularized` | region | Ireland | 193 | 92 | 0.611570 | 0.804348 | 0.195652 | 0.131725 |
| `xgboost_regularized` | region | London Region | 647 | 329 | 0.568138 | 0.899696 | 0.100304 | 0.131725 |
| `xgboost_regularized` | region | North Region | 368 | 162 | 0.558233 | 0.858025 | 0.141975 | 0.131725 |
| `xgboost_regularized` | region | North Western Region | 579 | 309 | 0.563147 | 0.880259 | 0.119741 | 0.131725 |
| `xgboost_regularized` | region | Scotland | 680 | 348 | 0.593385 | 0.876437 | 0.123563 | 0.131725 |
| `xgboost_regularized` | region | South East Region | 417 | 189 | 0.542373 | 0.846561 | 0.153439 | 0.131725 |
| `xgboost_regularized` | region | South Region | 645 | 261 | 0.506818 | 0.854406 | 0.145594 | 0.131725 |
| `xgboost_regularized` | region | South West Region | 478 | 211 | 0.541667 | 0.862559 | 0.137441 | 0.131725 |
| `xgboost_regularized` | region | Wales | 419 | 219 | 0.590778 | 0.936073 | 0.063927 | 0.131725 |
| `xgboost_regularized` | region | West Midlands Region | 511 | 256 | 0.552693 | 0.921875 | 0.078125 | 0.131725 |
| `xgboost_regularized` | region | Yorkshire Region | 409 | 214 | 0.596386 | 0.925234 | 0.074766 | 0.131725 |

## Contracts and operational status

| Model | Hyperparameters | Explanation | Status |
|---|---|---|---|
| `xgboost_regularized` | `{"base_score":null,"booster":null,"callbacks":null,"colsample_bylevel":null,"colsample_bynode":null,"colsample_bytree":0.8,"device":"cpu","early_stopping_rounds":null,"enable_categorical":false,"eval_metric":"logloss","feature_types":null,"feature_weights":null,"gamma":0.1,"grow_policy":null,"importance_type":null,"interaction_constraints":null,"learning_rate":0.03,"max_bin":null,"max_cat_threshold":null,"max_cat_to_onehot":null,"max_delta_step":null,"max_depth":5,"max_leaves":null,"min_child_weight":10,"missing":null,"monotone_constraints":null,"multi_strategy":null,"n_estimators":400,"n_jobs":1,"num_parallel_tree":null,"objective":"binary:logistic","random_state":42,"reg_alpha":0.05,"reg_lambda":5.0,"sampling_method":null,"scale_pos_weight":1.0,"subsample":0.8,"tree_method":"hist","validate_parameters":null,"verbosity":0}` | tree-shap-interventional-v1 | selected production contract; release approval not evaluated |
| `logistic_baseline` | `{"C":1.0,"class_weight":null,"dual":false,"fit_intercept":true,"intercept_scaling":1,"l1_ratio":null,"max_iter":2000,"multi_class":"deprecated","n_jobs":null,"penalty":"l2","random_state":42,"solver":"lbfgs","tol":0.0001,"verbose":0,"warm_start":false}` | linear-shap-interventional-v1 | fallback contract; release activation not evaluated |
| `random_forest_b` | `{"bootstrap":true,"ccp_alpha":0.0,"class_weight":"balanced_subsample","criterion":"gini","max_depth":null,"max_features":"sqrt","max_leaf_nodes":null,"max_samples":null,"min_impurity_decrease":0.0,"min_samples_leaf":5,"min_samples_split":2,"min_weight_fraction_leaf":0.0,"monotonic_cst":null,"n_estimators":500,"n_jobs":1,"oob_score":false,"random_state":42,"verbose":0,"warm_start":false}` | benchmark-only; no production explainer | benchmark challenger only |

## Decision

The repository selects `xgboost_regularized` as its production training contract; this comparison does not verify that a production release has been approved or activated.

Against Logistic on this validation artifact, XGBoost is better on PR-AUC, ROC-AUC, Top-20% precision, Top-20% recall, Brier score, ECE. Logistic remains the rollback contract.

Against Random Forest B, XGBoost is worse on PR-AUC, ROC-AUC, Top-20% precision, Top-20% recall, Brier score, ECE. Its benchmark payload is smaller, its measured throughput is higher, and its maximum eligible FNR gap is narrower. Random Forest remains benchmark-only because no production adapter/release contract is supplied to this command.

Promotion is a governance exception over validation evidence, not scientific proof of superiority. Activation requires a separately verified promotion-decision artifact; a new untouched temporal cohort is required to change the scientific evidence status.

## Source hashes

- `candidate_configurations.json`: `cb9328ef7e8dd575c34f3796828b13f9e832624f16afa0d3e35f0c34750c985f`
- `validation_metrics.csv`: `a0d6099cd6abde0f0bb071d662c768d018da53b0072bf3462d034f22b09a01fa`
- `validation_predictions.parquet`: `5603b304250d4999b336d96b1f1b4114671b75956897450bc9afc7bcc5b275ec`
- `validation_slice_metrics.csv`: `49fceff6675bba69c51d4d2092f14fad3364ebe8eb10bbab27fd034f6c77a96d`
