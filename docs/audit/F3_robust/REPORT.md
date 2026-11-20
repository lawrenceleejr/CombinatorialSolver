# F3 — robustness/invariance of r14A (written by main from the agent's final message; tables in out/tab/)
Run locally on a prescaled set (807 OSPool jobs never started and were removed); nominal reproduces quoted ε (700 P 0.00334 vs 0.0033).
Limit ratio = (B_var/B_nom)/(ε_var/ε_nom), f=10%; "re-tuned" = P cut moved so B_var = B_nom.

1 Code: ONNX = PyTorch (Δlogit p99 3e-4; no argmax/0.997 flips); scale/η-flip exact; φ-rot float noise; jets 8–12 permutation-invariant.
  Bugs (not hit by current pipeline): 5 jets → p=1, m_est NaN; ≤3 jets NaN; pT=0,pz≠0 jet → NaN; ONNX requires pT-sorted input
  (no check); φ→−φ not a symmetry (0.13% flips).
  NOT IR SAFE: adding one 0.001–10 GeV jet takes QCD pass at P≥0.997 from 0.2% to 44–57%; one 30 GeV jet ×3.8 QCD, ×4.1 signal.
2 Jet mass: zero masses (re-tuned) 400 ×1.7, 700 ×1.51±0.18, 700 P+all ×2.0, 1000 ×1.13, 1500 ×4.4, 2000 ×7.5.
  Smear 10% ≤×1.1; 20% ×1.37; 30% ×1.82; 20% smear + 10% JER ×1.50±0.23 at 700 (×1.34–1.51 at 1–2 TeV). Mass scale 0.9 ×1.41 (700).
3 JER +10% ×1.47 (700); threshold 50 GeV ×1.43; JES +5% ×1.15–1.3; +2 pile-up jets ×1.14–1.31 (fixed cuts up to ×1.7);
  η/φ resolution, |η|<2.4, 5% jet loss ≤×1.15.
4 Mass variable: only 39% of P-selected signal in [0.85,1.1]M (23% in [0.92,1.04]M); 68% width 0.57–0.60 M; m_est no better than m_avg.
  P efficiency peaks at H_T/M 2.3–3.0, ×4–5 lower outside.
5 P ≥0.997 passes 0.0005/0.013/0.045/0.18/0.35 of signal at 0.4/0.7/1/1.5/2 TeV (per-mass cuts mandatory).
  Tight P flattens QCD m_avg (×40 fall at P≥0.9 → flat at 0.997 → rising at 0.9985). High-mass fit χ²/ndf 47/12 near 2.4 TeV sample edge.
Ranked fragility: jet-mass resolution; IR unsafety / multiplicity calibration; JER; mass scale; QCD sculpting + H_T/M dependence; contract bugs.
Not verified: migration from P_nom<0.9 for large perturbations; JER/JES below 30 GeV; resolution-aware or massless retraining.
