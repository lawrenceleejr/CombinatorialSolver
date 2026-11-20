# PairNet r14A audit (Nov 2026): analysis scripts

One directory per audit piece; the findings are in `docs/audit/AUDIT.md` and `docs/audit/F*/`.
Data, caches and job outputs live on the access point (`/scratch/llee/pairnet/audit/F*`) and are not
in the repository.

| dir | content |
|---|---|
| F0_codereview | code review notes of model / loss / loader / training |
| F1_stats | limit pipeline re-implementation, corrections, bootstrap, CLs checks |
| F2_data | duplicates, generator filters, shower / tune / signal-model variants, CMS data anchor |
| F3_robust | invariances, jet-mass / JES / JER / pile-up / soft-jet robustness |
| F4_peak | ancestry-truth ceiling of signal in the mass peak |
| F5_train | seed / length / capacity / loss-variant trainings and the harvest script |
| F6_stage2 | pooled mass-parametrised stage-2 MLP and other analysis-level tests |
| F7_jetassign | free-size per-jet assignment network (r16J) generation and trainings |
| F8_capacity, F8_saturation | interpretation attention / width tests; loss decomposition |
| F9_data | data-scaling and per-class overtraining tests |
| F10_peak | signal and QCD in the peak, sculpting, ancestry targets, DisCo, decorrelated cut (ddt.py) |
