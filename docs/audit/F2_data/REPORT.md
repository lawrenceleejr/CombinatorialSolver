# F2 — data integrity and generator realism (written by main from the agent's final message)

Verdicts
1. Duplicates/split leakage: OK. 0 duplicates in 209 families, splits are file ranges; cross-family overlaps are the same files stored twice.
2. Metadata: OK. Identical detector emulation for signal and QCD; differences only in generator filters/pTHat/bias weights.
   No pile-up, no non-Gaussian tails, no q/g or b response differences in either. Truth before vs after FSR: <0.01 effect.
3. Filters: OK. H_T filter 100% efficient by construction; symmetry filter inefficiency <0.6% (95% CL);
   pTHat thresholds agree in overlaps (1.00±0.01 at P>0.9; final cut 1.01–1.12 ±0.09; one 2.6σ outlier 0.77±0.085 at H_T>1500).
   pTHat<80 not directly checked (0% share at P>0.9, likely negligible).
4a. Shower α_s 0.123/0.150: B scale 0.8/1.2, tight-P fraction stable; final cut too few events.
4b. CMS-EXO-21-004 anchor (with emulated q/g discriminator): MC/data ≈ 1.1, 0.55, 0.6 (±50–100%) — Pythia within ~×2, if anything low.
4c. Beyond 2→2+shower: unknown at final cut. CP5-like at P>0.99: 0.45±0.22; Vincia inconclusive.
5. Signal: FSR α_s ±0.005 → ±35% efficiency at M≤1 TeV (±15% at 1.5, ±5% at 2 TeV) — the tight P cut uses the radiation pattern.
   Decay model at 700 GeV: width 10% ×0.72; cascade via squark 0.5M ×2.05, 0.85M ×0.43; udb ×1.06. Limit holds for the flat 3-body benchmark only.

Jobs queued: 43043451 (alt showers/tunes QCD, 1050), 43043452 (pTHat 40–80 / flagged, 450), 43043455 (signal variants, 96).
Harvest: alt_cmp.py cp5|v7 'jobs_out/<prefix>_c*.npz' [1500]; sv_cmp.py jobs_out 1500 0.998 nom w5 w10 ch50 ch85 udb;
QGD_REP=30 cms_emul.py cms_s2.npz 3 jobs_out/s2_c*.h5 && cms_cmp.py cms_s2.npz noqgd qgd
Not verified: final-cut B under other generators/ME merging, pTHat<80 directly, signal under CP5/Vincia, real b/q-g response, pile-up.
