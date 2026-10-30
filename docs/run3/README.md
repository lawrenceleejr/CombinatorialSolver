# Run 3 gluino → 3 jets search with PairNet r14A: current state (Oct 29, 2026)

- **Model:** `models/pairnet_gated_12jet_win5_lowmass_boost_r14A.onnx` (+ `.json`), PyTorch checkpoint and validation in `models/r14A/`.
- **Write-up:** `RUN3.md` (full history of the study and the arguments).
- **Recommended use:** r14A pairing → A < 0.1 → m_avg window → P(signal) ≥ P_min(M), 1 − P_min ≈ 10^(−2.03 − 0.51 M/TeV),
  boost cuts (H_T/M, Δ) below 1 TeV, 2–3 P(signal) categories. Per-mass cuts: `results/working_points.json`.
- **Projected sensitivity** (300 fb⁻¹, 13.6 TeV, mass window, 10% on B, categories): reach ≈ 2.2 TeV; CMS Run 2 (observed)
  already excludes 200–1780 GeV (`figures/deck_excl.png`, `results/window10_enriched_tables.md`).
- **Around 700 GeV:** 118 → 103 fb with targeted MC, 91 fb with the window [0.92, 1.04] M; 71 / 64 fb with 5% on B
  (`figures/compare700.png`, `results/compare700.json`).
- **Background uncertainty scan:** `figures/fsys_scan2.png`, `results/fsys_scan.json`.
- **Extra kinematic variables at 700 GeV:** `results/extra_variables_700GeV.txt` (none adds purity except extra-jet activity, ≈ ×1.3).

Scripts: `scripts/run3_bumphunt/` (selection tuning, enriched/targeted QCD merge, limits, plots, OSPool job files) and
`scripts/pairnet_study/` (the rest of the study toolkit; paths point at the OSPool areas used for the study).
Slides: the "PairNet r14A: use and projected sensitivity" deck (claude.ai artifact HwLsPFYWtKfJKeHghta1Ep).
