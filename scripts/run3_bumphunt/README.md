# Run 3 bump-hunt study (Oct 22–23, 2026)

Low-mass and high-mass expected limits for a gluino → 3 jets bump hunt on a smoothly falling m_avg spectrum, with
PairNet r14A on held-out MC. These are study scripts: input paths point at the OSPool scratch/OSDF areas used for the
study (`/scratch/llee/pairnet`, `/ospool/uc-shared/project/futurecolliders/llee/pairnet`).

| script | what it does |
|---|---|
| `lowmass_study4.py` | 200 GeV–1 TeV: tunes P(signal), H_T/M, Δ, Dalitz, p_T,6 and A cuts per mass on one half, evaluates on the other. `FOM=fisher` uses the profiled shape-fit sensitivity (signal + dijet background with free parameters, `NPAR=4/5`); `PEAK_MAX` requires the QCD m_avg peak ≤ PEAK_MAX·M; `CATS=n` adds P(signal) categories; `SEED_GRID` joint scan of the boost cuts; `FSYS_JSON` spurious-signal-like term. |
| `fisher_high.py` | 500 GeV–2.4 TeV, same figure of merit on the high-mass samples (per-mass P cut, categories). |
| `exclusion_fit.py` | Exclusion plot: shape fit (4-/5-parameter, categories), window counting, Run-2-like counting, CMS-EXO-21-004 × √(128/300). |
| `bump_plot.py` | Asimov pseudo-data with signal at 5σ_μ, background-only refit, residuals. |
| `lever_arm.py` | Shape-fit sensitivity vs start of the fit range for a fixed selection. |
| `peak_fom.py` | QCD peak position after the best selection for the counting and shape-fit figures of merit. |
| `peak_anatomy.py`, `peak_info.py` | Signal m_avg by truth category (correct / wrong split / unmatched) and its share of the fit information. |
| `spurious.py`, `spurious_compare.py` | Spurious-signal test of the fit function on the QCD MC. |
| `spectra_fit.py` | m_avg spectra with a dijet fit from the peak. |
| `limits.py` | Asimov counting limit with background uncertainty. |

Results and the written argument: `~/pairnet_study/RUN3.md` (section "Bump hunt on a smoothly falling spectrum").
