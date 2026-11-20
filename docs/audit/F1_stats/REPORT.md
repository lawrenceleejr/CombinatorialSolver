# F1 — statistics and limit pipeline (written by main from the agent's final message)
pipe.py reproduces the published numbers exactly (700 P+all 151.5045, +cats 117.5302; 400 3173.76; all high-mass).

Corrected expected limits (fb, 300/fb, f=10% on B)  [factor >1 = published optimistic]
M     pub single  corr single  factor  MC σ(log)  pub cats  corr cats (worst ρ)  factor
400   3103        4030         1.30    0.16       2486      4170                 1.68
700   151.5       168          1.11    0.17       117.5     168                  1.43
1000  30.3        32.0         1.05    0.12       21.4      29.1                 1.36
1500  7.69        8.09         1.05    0.20       6.41      8.09                 1.26
2000  1.18        1.20         1.02    0.13       1.02      1.17                 1.15
Corrections: falling-spectrum requirement on all MC; configuration chosen on tuning half; Gaussian constraint; categories with worst ρ.
Reach: single 2123 published → 2113 (Gaussian), 2135 (log-normal), MC spread 2101–2215; categories 2197 → 2129 (worst ρ).

1 limits.py: s_up = on/off Asimov; vs Poisson/toys within 2–7%. Uncorrelated per-category 10% is optimistic; "fully correlated" is not
  conservative either (common nuisance constrained by S/B differences). Weakest at ρ≈0.5–0.7: ×1.34/1.31/1.31/1.25/1.14.
2a best-of choice on evaluated limits: ×1.00–1.05. 2b evaluation half unbiased but MC-limited: N_eff 11–48 (8–15 at 1.9–2.3 TeV),
  single events carry up to 24–33% of B. 700 GeV with 2.2× tail MC gives 134.4 (not a lucky fluctuation).
2c BUG: falling-spectrum (PEAK_MAX) requirement checked on tuning half only → 400 ×1.13, 700 ×1.08, 1000 ×1.03.
3 QCD merge OK (ht/vlm 0.98–1.04; tail/base 0.99–1.00). vlm complete only for m_avg<1150 but merged at all masses (14% low above; no
  effect on curves). old/vlm 0.926±0.018 at H_T 900–950 (0.4% on B).
4 Leakage OK (held-out/train tail efficiency ≈1.0±0.06).
5 Signal normalisation OK. Unresolved: power-shower samples keep 0.38 of tried events vs 0.93 (×0.43 window efficiency at 700).
  CMS curve: acceptance division correct; √(128/300) conservative ≤1 TeV, generous up to ×1.5 at ≥1.5 TeV; no 13→13.6 correction.
  CMS Run-2 expected beats this projection below ~1700 GeV (700: 93 vs 152; 1000: 12.5 vs 30.3); PairNet better only ≥1.7–2 TeV.
6 window10.py docstring says "fully correlated" but headline is uncorrelated; fisher_high.py KeyError for other tails;
  cfg.json/suite scripts point to deleted /tmp scratchpad paths.
