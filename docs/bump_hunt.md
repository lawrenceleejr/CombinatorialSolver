# From combinatorial solver to an LHC bump-hunt search

This note explains how the pieces in this repo fit together into an actual
dijet-of-trijets bump-hunt search (pair-produced RPV gluinos → 6 jets, possibly
+ ISR), how the setup is made robust to signals with **resonant structure inside
the jet triplet**, how to use the **cut / ROC / bump-hunt tooling**, and a
roadmap for building the **statistical search** on top of it.

---

## 1. The analysis in one sentence

The network assigns the 7 leading jets into (ISR, triplet₁, triplet₂); we
reconstruct the two triplet masses, and the **average candidate mass
`m_avg = (m₁+m₂)/2`** is the bump-hunt observable. A pair-produced resonance of
mass `M` appears as a peak at `m_avg ≈ M` on top of a smoothly-falling QCD
multijet spectrum.

Everything downstream is about (a) making `m_avg` reconstruct the right peak for
*any* signal topology and (b) shaping the QCD spectrum so the peak is visible
and statistically interpretable **without sculpting** the background.

---

## 2. Robustness to resonant structure inside the triplet  (`g̃ → q + q̃(→qq)`)

A direct RPV decay `g̃ → qqq` populates the **interior** of the Dalitz triangle
(flat 3-body phase space). A **cascade** `g̃ → q + q̃`, with an on-shell squark
`q̃ → qq`, instead places the squark mass at a **fixed pairwise mass** — a
populated **band/line** in the Dalitz triangle. Both are signal, and both must
be found.

Two facts make the existing setup robust to the cascade:

1. **Truth labelling is topology-agnostic.** Labels are
   `argmin |m₁−m₂|` over all interpretations (`dataset.py`,
   `use_mass_asymmetry_labels`). The two gluinos are equal-mass, so the correct
   3+3 split still minimises `|m₁−m₂|` regardless of what happens *inside* each
   triplet. The cascade is grouped correctly and `m_avg` peaks at the gluino
   mass. (Verified: a 1 TeV gluino with a 400 GeV squark reconstructs a sharp
   1000 GeV peak — see `scripts/generate_mock_data.py --cascade`.)

2. **The QCD-rejection features do not assume flat phase space.** The intra-group
   features (Lund `kT`/`z`, ECF₂/ECF₃/D₂, Dalitz pairwise masses) tag the
   *soft/collinear* QCD limit, which lives at the Dalitz **edges/corners**
   (one pairwise mass → 0). A squark band sits at a *finite* pairwise mass in
   the interior, so it is **not** in the region these features penalise — unless
   the squark is light enough to sit near an edge.

That last caveat is exactly why the Dalitz cut is implemented as **two scannable
shapes** (`src/cut_analysis.py`):

| Cut | Keeps events with | Removes | Effect on cascade |
|-----|-------------------|---------|-------------------|
| `dalitz_corner` | `median(d₁,d₂,d₃) > t` | only the 3 **corners** (two pairs soft/collinear) | safe — preserves bands anywhere in the interior |
| `dalitz_edge`   | `min(d₁,d₂,d₃) > t`    | a **band along all 3 edges** | aggressive — bites into a *light*-squark band near an edge |

where `dᵢ = m²(pair)/M²` are the three normalised pairwise masses (∑dᵢ ≈ 1), and
each metric is taken as the **minimum over the two triplets** (an event is QCD-like
if *either* parent is edge/corner-like). Use the ROC scan to choose `t` per signal
model: for a heavy squark, `dalitz_edge` buys extra QCD rejection for free; for a
light squark, prefer `dalitz_corner` (or leave the Dalitz cut off).

> **Action when a real cascade MadGraph sample exists:** train with it mixed into
> the signal (it already labels correctly), then re-run `src/analysis.py` and
> confirm the `m_avg` peak and the Dalitz-cut efficiencies on it.

---

## 3. The cut / ROC / bump-hunt tooling

The requested selection — **Δφ > 2.5, mass asymmetry < 0.4, average boost < 2**,
plus the Dalitz edge/corner cuts — and the "what do I gain" study are implemented
once in `src/cut_analysis.py` (pure-numpy logic) + `src/cut_plots.py` (figures),
and surfaced in two places that are **guaranteed to agree**:

### a) Standalone, on the final network (ONNX or checkpoint)
```bash
python -m src.analysis \
    --onnx onnx_snapshots/final_.../ml_model_final_....onnx \
    --signal "data/sig*.h5" --background "data/qcd*.h5" \
    --output results/bump_hunt \
    --dphi-min 2.5 --asym-max 0.4 --boost-max 2.0 \
    --dalitz-corner 0.05          # optional; --dalitz-edge for the band cut
# Or straight from a checkpoint (no ONNX needed):
python -m src.analysis --checkpoint checkpoints/best_model.pt --signal ... --background ...
```
Outputs (in `--output`):
- `cutflow.csv` — **fraction of signal and background remaining** at each cut, with running `S/√B`.
- `bump_hunt_summary.pdf` — four panels:
  1. **ROC single-cut scans** (signal eff vs background rejection, one curve per variable, with AUCs);
  2. **combined working-point cloud** — *many* cut combinations scattered in (sig eff, bkg rej), with the **Pareto front** and the **nominal working point** marked;
  3. **cutflow bars** (signal/background fraction remaining);
  4. the **`m_avg` bump-hunt histogram** before (faint) vs after (bold) cuts.
- `observables.npz` — per-event arrays for custom re-plotting.

`--input-norm {auto,ht,none}` controls input scaling (ML model wants HT-normalised
inputs, the classical solver wants physical units; `auto` infers from the filename).

### b) Animated by epoch during training
When a QCD sample is supplied (`--qcd-data`), training writes
`plots/bump_hunt_roc_anim_latest.gif`: per epoch, the **ROC single-cut scans** and
the **combined working-point cloud** (with the nominal point), so you watch the
achievable trade-off — and the signal/background kept at the nominal cuts — improve
as the network learns. It reuses the exact same `cut_analysis`/`cut_plots` code as
the standalone tool.

---

## 4. Building the statistical search

The combinatorial solver + cuts give you a clean `m_avg` spectrum. A *search*
needs a background estimate and a statistical interpretation. Recommended order:

### 4.1 Background estimation (the crux of a bump hunt)
QCD multijet is hard to predict from MC, so estimate it **from data**:

- **Functional-form fit.** Fit a smoothly-falling parametric form (the classic
  dijet `f(x) = p₀(1−x)^{p₁} x^{p₂+p₃ln x}` with `x = m_avg/√s`, or a
  `(1−x)^a / x^b` family, or a 3–5 parameter Bernstein/`Dijet-N`) to the `m_avg`
  spectrum, fit excluded in the signal window, then look for a localised excess.
- **Sliding-window scan / BumpHunter.** Scan windows across `m_avg`, compute local
  p-values from the side-band-predicted yield; correct for the **look-elsewhere
  effect** over the scanned mass range.
- **ABCD / data-driven transfer.** Define two as-decorrelated-as-possible
  variables (e.g. `mass_asym` and a Dalitz/NN-score variable), form four regions,
  predict the signal region from the other three. Requires the two variables to be
  approximately independent **for background** — check with the tooling below.

### 4.2 Decorrelation / no-sculpting validation (do this *before* limits)
A bump hunt is only valid if the **selection does not carve a fake bump** into the
QCD `m_avg` shape. The network is explicitly built to avoid this (HT-normalised
inputs, gradient-reversal mass adversary, `lambda_bg` pushing QCD to low-mass/high-
asymmetry rather than to a mass *window*). **Verify it:**
- In `bump_hunt_summary.pdf`, overlay the QCD `m_avg` **shape** before vs after cuts
  (normalise to unit area) — it should stay smooth and monotonic, not grow a bump.
- Scan each cut and confirm the QCD shape is stable (a "bump-vs-cut" stability plot
  is a natural next addition to `cut_plots.py`).
- For ABCD, plot QCD `m_avg` in each of the four regions and check the shapes agree
  (the closure test).

### 4.3 Statistical interpretation
- Build templates (signal from MC at each mass point, background from the fit /
  ABCD) and run a **profile-likelihood** fit. `pyhf` (HistFactory) is the natural
  choice and is pip-installable; the `m_avg` histogram + uncertainties feed it
  directly.
- **Signal-injection tests:** inject signal at known cross-sections and confirm you
  recover the injected μ (bias/coverage of the fit).
- **Expected & observed limits** (CLs) and **discovery significance** (`q₀`,
  asymptotic formulae) as a function of the probed mass — this is the headline
  result.
- **Mass scan + LEE:** repeat across the `m_avg` range; convert the most
  significant local excess to a global p-value (pseudo-experiments or the
  Gross–Vitells method).

### 4.4 Systematics
- **Detector:** jet energy scale / resolution. The training `pt_smear_frac` is a
  resolution proxy; propagate JES/JER by shifting/smearing inputs and re-running
  `src/analysis.py` to get up/down templates.
- **Background model:** the dominant systematic in a bump hunt is the **fit-function
  choice** → estimate a *spurious-signal* uncertainty (inject zero signal, fit with
  the nominal and alternative functions, take the fitted yield as the systematic).
- **Theory:** signal cross-section (scale/PDF) for the limit interpretation.

### 4.5 Practicalities
- **Trigger & preselection** (multi-jet HT or single-jet triggers; `n_jets ≥ 6/7`,
  jet `pT`/`η` fiducial cuts) and the resulting acceptance.
- **Blinding:** keep the signal window blinded until the background model and
  systematics are frozen.
- **Normalisation/weights:** real samples carry per-event weights
  (`EventVars/normweight`, `event_features[:,6]`). `Observables` already supports a
  `weight` array; wiring weights through `src/analysis.py` (read `normweight` aligned
  to the kept events) is the first thing to add for absolute yields.

---

## 5. Suggested next steps in this repo
1. Wire per-event **weights** from the HDF5 into `src/analysis.py` for true yields.
2. Add a **QCD-shape-stability / sculpting** panel to `cut_plots.py` (QCD `m_avg`
   shape vs cut tightness).
3. Add an **ABCD closure** helper (pick two variables, print the four-region
   prediction and closure).
4. Add a thin **`pyhf` template-fit** script that consumes `observables.npz` and a
   background template to produce a limit/significance vs mass.
5. When a real cascade (`g̃→q+q̃→qq`) MadGraph sample lands, mix it into training and
   re-validate the `m_avg` peak and Dalitz-cut efficiencies with `src/analysis.py`.
