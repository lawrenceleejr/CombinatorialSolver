# F5 — training experiments (written by main from the agent's final message; harvest: python harvest.py → TABLE.md)
a. Sweep noise: no configuration was ever re-run with only the seed changed. Bootstrap test error at 0.5% QCD ≈ ±0.002;
   epoch scatter 0.001–0.01. Surviving claims: window term (r8 vs r5D, ~8σ); r15 tail-QCD worse than r14A (−1.2±0.1 pts at 0.5%).
   Not established: 12 vs 10 vs 16 jets (equal within 1.5σ); "capacity does not help" (only tested at 10 jets/20 epochs).
b. r14A: val loss still falling at the last epoch (2.1743, 2.1728, 2.1718), train-val gap 0.004 → under-fitted, schedule-limited.
   AUC flat 0.930–0.933; last epoch selected. Checkpoint selection (win_a01) never looks at the gate.
r14A reference (signal eff ×100 in window, A<0.1; QCD eff = weighted fraction of all test QCD with A<0.1 and gate above threshold)
QCD eff  neff   M400  M700  M1000 M1500 M2000
5%       17266  21.6  33.1  37.0  46.7  59.6
0.5%     2161   9.7   21.3  27.3  39.3  53.9
0.1%     522    3.6   12.2  18.7  30.8  46.8
0.02%    139    1.0   5.7   11.2  22.5  38.9
Submitted (OSPool GPU, idle): f5_s1..3 (43043447–49), f5_e30, f5_drop0, f5_big6, f5_big8, f5_big6tail, f5_wp, f5_phys, f5_focal,
f5_maug, f5_m0. Branch audit/f5-train (ba89394), repo tarball stage/repo_f5.tar.gz.
