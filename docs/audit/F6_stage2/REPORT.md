# F6 — analysis-level headroom with r14A fixed (written by main from the agent's final message; tables in out/tables.md)
Nested: everything fixed on one parity half, evaluated once on the other, geometric mean of both; stage-2 cross-fitted inside
the tuning half; paired Poisson bootstrap (R=200) of the evaluation half only (tuning variance not included, ±5–10% likely).
Reference = P+all+cats under the same nesting.
1 Pooled mass-parametrised MLP stage-2 (16 features in units of m_avg + log m) + 3 score categories:
  limit ratio to current: geometric mean 0.67 (f=0.10), 0.71 (f=0.05); 0.35–0.84 per mass, beyond bootstrap error.
  gbdt 0.94 (fails shape requirement at 300–350); 2-D density ratio 0.84; lin2 (logit P + a·log HT/m) 0.90.
2 Hypothesis-conditioned pairing: worse below 1.5 TeV (sculpts QCD into window); not recommended.
3 m_est as search variable: worse or equal.  4 Pairing-distribution variables: within noise.
5 Window-edge re-tune + cats: 5–8%, comparable to tuning noise.
Recommended: mlp_core + 3 categories. New limits f=0.10 (fb): 300 6.2e3, 400 1.96e3, 500 470, 700 95, 1000 16.3, 1500 2.7,
2000 0.65, 2400 0.26.
Caveats: gain relies on QCD-MC modelling of H_T/m and event shapes; categories treated with uncorrelated 10% (F1: optimistic);
low mass MC-limited (top-category neff 16–24 at 450/500). fisher_high.py has no fileclass cut for signal (training files may enter
published high-mass limits). Nested reference up to 2× below published at 1100–1600.
