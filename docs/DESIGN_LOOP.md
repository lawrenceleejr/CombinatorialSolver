# Design loop: autonomous sample generation and model testing

Operating instructions for a Claude Code session that has been handed compute
to improve the CombinatorialSolver network. The session reads this file first,
then `experiments/QUEUE.md` and `experiments/ledger.csv`, and works through the
queue one experiment at a time until the budget in the handoff is spent.

Related files:

| file | purpose |
|---|---|
| `CLAUDE.md` | short project context, auto-loaded by every session |
| `experiments/QUEUE.md` | ordered experiment queue with hypotheses and success criteria |
| `experiments/ledger.csv` | one row per experiment, the single source of truth for results |
| `experiments/TEMPLATE_report.md` | per-experiment report skeleton |
| `data/MANIFEST.md` | every generated sample: producer commit, config, seed, validation |

---

## 0. Handoff

The user starts a loop by pasting a block like this. Missing fields take the
defaults in brackets.

```
Run the design loop in docs/DESIGN_LOOP.md.
compute:      Mac Studio, Apple silicon GPU via MPS   [detect with the checks in section 2.1]
gpu_hours:    24                                      [8]   (wall-clock hours of the Mac's GPU)
cpu_cores:    16                                      [all but two]
disk_gb:      200                                     [100]
deadline:     2026-10-01T09:00Z                       [none]
docker:       Docker Desktop with Rosetta             [detect]
network:      yes                                     [detect]
phase:        A                                       [A]   (A = Tier 0 fixes, B = showered data, C = topology head)
start_at:     E0                                      [first item in QUEUE.md with status todo]
stop_after:   E5                                      [when budget is spent]
branch:       claude/design-loop-<date>               [claude/design-loop-<date>]
```

Anything not covered by these fields or by this document is the session's
call, within the guardrails in section 8.

---

## 1. Ground rules

1. **Commit and push before every long run.** Nothing that takes longer than
   ten minutes starts on uncommitted code. Push to the loop branch, never to
   `main`.
2. **One change per experiment.** If two changes are needed to test a
   hypothesis, that is two experiments or an explicit ablation with both
   arms run.
3. **Frozen definitions.** Metric definitions (section 4), the test split
   (section 3.4) and the signal-region definition do not change inside a
   loop. If one must change, the loop ends, the change is recorded in this
   file, and every previous ledger row is marked `superseded`.
4. **Never touch the test split** until the final evaluation of an
   experiment. Model selection uses the validation split only.
5. **Always compute the classical baseline** on exactly the same events and
   smearing realisation as the model, and report the difference with an
   uncertainty. A model number without its baseline is not a result.
6. **Three seeds before a keep decision** whenever the primary metric moved
   by less than three times its single-run uncertainty.
7. **Every plot follows Tufte.** No gridlines, no boxes, no chartjunk, no
   legends when direct labels will do. Data ink first. PDF output. Serif type
   via the repo's existing `_init_plot_style` and `_style_axis` helpers in
   `src/train.py`. Small multiples rather than overlays above three series.
   Uncertainty shown as a band or as error bars, never omitted.
8. **Report faithfully.** A run that diverged, crashed, or ran out of budget
   is recorded as `failed` with the reason. No result is ever inferred from a
   run that did not finish.
9. **Budgets bind.** Per-experiment caps in `QUEUE.md`; total from the
   handoff. Kill a training run that exceeds its cap or that sits below the
   classical baseline after 30% of its cap.
10. **Clean up.** Delete checkpoints except the best one per experiment,
    delete intermediate LHE and HepMC files after the HDF5 is validated.

---

## 2. Environment bring-up (start of every loop)

The reference machine is a Mac Studio with an Apple silicon GPU. PyTorch
reaches it through the MPS backend, which `get_device` in `src/utils.py`
already prefers. A Linux box with an NVIDIA GPU works with the notes in
section 2.6.

Run every check, record the output in `experiments/ENV_<date>.md`, and fix
anything that fails before starting an experiment.

### 2.1 Checks (macOS)

```bash
sw_vers                                                  # macOS version
sysctl -n machdep.cpu.brand_string                       # chip
sysctl -n hw.memsize | awk '{print $1/1e9 " GB unified memory"}'
system_profiler SPDisplaysDataType | grep -E "Chipset|Cores"
python3 --version                                        # 3.10+
python3 -c "import torch; print(torch.__version__, 'mps:', torch.backends.mps.is_available())"
python3 -c "import h5py, numpy, yaml, matplotlib; print('ok')"
docker --version && docker info >/dev/null 2>&1 && echo docker-ok || echo no-docker
df -h .                                                  # disk
git status --short && git log --oneline -1               # clean tree on the loop branch
```

### 2.2 Installs (macOS)

Use an arm64 Python from Homebrew or miniforge. The default PyPI torch
wheel for arm64 macOS includes MPS.

```bash
pip install -r requirements.txt
python3 -c "import torch; x = torch.ones(1, device='mps'); print((x * 2).item())"
```

Set `PYTORCH_ENABLE_MPS_FALLBACK=1` in every shell that runs training, so
an operator MPS lacks falls back to CPU instead of raising. If a run dies
with an MPS out-of-memory message, set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`
and halve the batch size; record both in the ledger row.

### 2.3 Sample producer (macOS)

The producer's Docker image is x86-64 only. Two routes; try A first.

**A. Docker Desktop with Rosetta.** In Docker Desktop, Settings > General,
enable "Use Rosetta for x86_64/amd64 emulation on Apple Silicon". In
Settings > Resources give Docker at least 8 CPUs and 16 GB. Then the
producer's normal commands work, slower than native. Clone next to this
repo, pin the commit in `data/MANIFEST.md`, and smoke-test with 100
parton-level events. The first call builds the image.

```bash
git clone https://github.com/lawrenceleejr/MadGraphMLProducer ../MadGraphMLProducer
cd ../MadGraphMLProducer && git rev-parse HEAD
./run -c configs/examples/gluino_rpv_1tev_uds.yaml -n 100 --shower off -o /tmp/smoke.h5
```

**B. Native MadGraph.** Only if the emulated image fails or is too slow.
`brew install gcc` for gfortran. Download the MadGraph5_aMC@NLO 3.5.x
tarball, put `mg5_aMC` on PATH, and inside it run `install pythia8` and
`install lhapdf6`. Install the RPVMSSM_UFO model into its `models/`
directory and run `convert model` on it, exactly as the producer's
Dockerfile does. Then `pip install pylhe pyhepmc pyjet pydantic jinja2
pyyaml click tqdm awkward vector` and run `python run.py --no-docker ...`
from the producer. Two caveats: the native runner has no `--shower off`,
so Tier 0 natively needs that flag added to `run.py` by mirroring
`run_docker.py`; and the shower step needs the `pythia8` Python module
importable from the same interpreter, which means building Pythia8 with
its Python interface.

Time the 100-event run and record the rate. Scale all later generation
estimates from it. Do not guess generation times.

### 2.4 Training smoke test

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.train --config configs/default.yaml --data "data/*.h5" 2>&1 | head -60
```

Expect `Using device: mps`. Record the seconds per epoch: it sets the
GPU-hour unit for every budget in `QUEUE.md`.

### 2.5 Unattended operation on the Mac

- **Keep the machine awake.** System Settings > Energy (or Displays >
  Advanced): enable "Prevent automatic sleeping when the display is off".
  Prefix every long job with `caffeinate -i`:

  ```bash
  caffeinate -i nohup python -m src.train --config configs/exp/E0.yaml --data "data/raw/tier0/*.h5" \
      > experiments/E0/train.log 2>&1 &
  ```

- **Run Claude Code inside tmux** so the session survives a closed window:
  `tmux new -s loop`, then `claude` in the repo. Reattach with
  `tmux attach -t loop`.

- **Permissions.** An unattended loop stalls at the first permission
  prompt. Claude cannot write its own permission settings, so before
  starting a loop the user creates `.claude/settings.json` in the repo
  with the block below, once. The alternative is to start with
  `claude --permission-mode acceptEdits` and accept the Bash prompts that
  appear in the first few minutes, saving each pattern to the allowlist
  from the prompt. The deny list blocks force pushes, pushes to `main`,
  hard resets, `sudo`, and destructive `rm`. Section 8 still applies.

  ```json
  {
    "$schema": "https://json.schemastore.org/claude-code-settings.json",
    "env": { "PYTORCH_ENABLE_MPS_FALLBACK": "1" },
    "permissions": {
      "additionalDirectories": ["../MadGraphMLProducer"],
      "allow": [
        "Read",
        "Edit(src/**)", "Edit(configs/**)", "Edit(scripts/**)",
        "Edit(experiments/**)", "Edit(docs/**)",
        "Edit(data/MANIFEST.md)", "Edit(data/splits/**)",
        "Edit(CLAUDE.md)", "Edit(README.md)", "Edit(requirements.txt)", "Edit(.gitignore)",
        "Bash(python *)", "Bash(python3 *)", "Bash(pip *)", "Bash(pip3 *)",
        "Bash(git status*)", "Bash(git log*)", "Bash(git diff*)", "Bash(git add *)",
        "Bash(git commit *)", "Bash(git checkout *)", "Bash(git switch *)",
        "Bash(git branch*)", "Bash(git merge *)", "Bash(git fetch *)", "Bash(git pull *)",
        "Bash(git push -u origin claude/*)", "Bash(git push -u origin exp/*)",
        "Bash(git push origin claude/*)", "Bash(git push origin exp/*)",
        "Bash(git rev-parse *)", "Bash(git stash*)",
        "Bash(docker *)", "Bash(./run *)", "Bash(../MadGraphMLProducer/run *)",
        "Bash(nohup *)", "Bash(caffeinate *)", "Bash(tmux *)",
        "Bash(tail *)", "Bash(head *)", "Bash(cat *)", "Bash(ls *)", "Bash(wc *)",
        "Bash(grep *)", "Bash(find *)", "Bash(du *)", "Bash(df *)", "Bash(ps *)",
        "Bash(kill *)", "Bash(mkdir *)", "Bash(cp *)", "Bash(mv *)",
        "Bash(rm -f checkpoints/*)", "Bash(rm -f logs/*)", "Bash(rm -f plots/*)",
        "Bash(rm -rf work/*)",
        "Bash(sw_vers*)", "Bash(sysctl *)", "Bash(system_profiler *)",
        "Bash(uname *)", "Bash(which *)", "Bash(date*)", "Bash(sleep *)"
      ],
      "deny": [
        "Bash(git push --force*)", "Bash(git push -f *)",
        "Bash(git push origin main*)", "Bash(git push -u origin main*)", "Bash(git push * main)",
        "Bash(git reset --hard *)",
        "Bash(rm -rf /*)", "Bash(rm -rf ~*)", "Bash(rm -rf data/*)",
        "Bash(sudo *)"
      ]
    }
  }
  ```

- **Disk.** Generated samples go under `data/raw/`, which git ignores.
  Check `df -h .` before every generation batch; stop generating below
  20 GB free.

### 2.6 Linux with an NVIDIA GPU

Replace the hardware checks with `nvidia-smi`, install torch from the
CUDA index (`pip install torch --index-url https://download.pytorch.org/whl/cu121`),
run Docker natively without Rosetta, and drop `caffeinate`. Everything
else in this document is unchanged.

---

## 3. Data

### 3.1 Tiers

| tier | what | when to use |
|---|---|---|
| 0 | parton level, no shower, pT smearing applied in the loader | fast iteration on labels, masking, architecture |
| 1 | Pythia8 shower, anti-kT R=0.4 truth jets, parametrised detector smearing in the loader | the first physically meaningful numbers; all topology-head work |
| 2 | Tier 1 plus Delphes | only after Tier 1 results are stable and if a Delphes image is available |

Tier 0 exists in `data/` today. Tiers 1 and 2 require the producer fixes
below.

### 3.2 Required producer fixes before generating training data

These were established by reading the producer at commit `365ae83`
(2026-09-23). Verify each is still needed before implementing; the producer
may have moved.

1. **Parton-level TARGETS are positional and wrong with an extra parton.**
   `run_docker.py::process_lhe_to_hdf5` writes g1 = final-state partons
   0,1,2 and g2 = 3,4,5. With `extra_partons: 1` the extra gluon is index 0,
   so every 7-parton event is mislabelled. Fix: derive g1 and g2 from the LHE
   mother links (each gluino is a status 2 particle; its three status 1
   daughters form one group). Write `parent_idx` (0 or 1, else -1) into
   `jet_features` as well.
2. **Showered output has no truth grouping.** `hdf5_writer.py` writes
   `parent_pdg` and `is_signal` but not `parent_idx`, and writes no
   `TARGETS` or `INPUTS/Source` groups. Fix: add `parent_idx` to
   `JET_FEATURES`, and write `TARGETS/g1`, `TARGETS/g2`, `INPUTS/Source/*`
   in the same layout as the parton path.
3. **Jet-to-parton matching is too loose.** `truth_extractor.py` flags a jet
   as signal if any final-state hadron descended from a gluino lies within
   R of the jet axis, first match wins. Fix: match each of the six decay
   quarks (the status 1 daughters in the LHE, or the status 23 quarks in the
   HepMC record) to its nearest jet within ΔR < 0.4, one jet per quark, and
   record which gluino. Jets that receive two quarks from the same gluino
   are merged decays; record them so the loader can mark the event as not
   fully reconstructable.
4. **No QCD configuration exists.** Add `configs/examples/qcd_multijet.yaml`:
   model `sm`, process `p p > j j`, `extra_partons: 2`, MLM matching with
   `xqcut: 30`, `qcut: 45`, and HT slices via the run card `htjmin`
   (for example 600, 1000, 1500 GeV) so the tail is populated. Keep the
   per-event weight in `EventVars/normweight` so slices can be combined.
   QCD files must have no `TARGETS` group and `is_signal` all zero.

Each fix is a commit to a fork or branch of the producer. Record the commit
hash in `data/MANIFEST.md`. Do not train on a file produced before the fix
that the experiment depends on.

### 3.3 Generation recipes

Signal grid, √s = 13.6 TeV, `go > u d s`, `extra_partons: 1`:

| mass (GeV) | Tier 0 events | Tier 1 events |
|---|---|---|
| 600 | 200k | 100k |
| 800 | 200k | 100k |
| 1000 | 200k | 100k |
| 1250 | 200k | 100k |
| 1500 | 200k | 100k |
| 2000 | 200k | 100k |

QCD, Tier 1 only: 500k events across the HT slices, weighted.

Commands are the producer's `./run` with `-c`, `-n`, `-s`, `-o`, and
`--shower off` for Tier 0. One seed per file, seed recorded in the file
name: `data/raw/tier<k>/<config>_m<mass>_s<seed>.h5`. Generate in chunks of
at most 50k events so a crash loses little, then concatenate at load time.

If the measured rate makes the table above exceed 20% of the loop's
wall-clock, scale every row down by the same factor and record that in the
manifest. Never reduce only the QCD sample.

### 3.4 Validation of every generated file

Run before the file is used, record the output in `data/MANIFEST.md`:

- jet multiplicity distribution (counts per n_jets);
- for signal: fraction of events with six uniquely matched jets; for those,
  fraction with both m(g1) and m(g2) within 10% of the generated mass
  (parton level must be above 99.9%; showered above 90% or investigate);
- for QCD: `TARGETS` absent, `is_signal` all zero, weight sum per slice;
- no NaN, no negative pT, φ in [−π, π].

A file that fails validation is deleted and regenerated or the producer is
fixed. It is never used.

### 3.5 Splits

Deterministic per file: event index hashed with a fixed salt, 80% train,
10% validation, 10% test. Split index arrays stored under
`data/splits/<file>.npz` and committed. The test split of every file is
read only by `src.evaluate` at the end of an experiment.

### 3.6 Detector proxy for Tiers 0 and 1

Applied in the loader with a fixed seed for validation and test, a fresh
seed per epoch for training:

- pT: σ/pT = 0.5/√pT ⊕ 0.03 with pT in GeV, clipped to [0.5, 1.5];
- η and φ: Gaussian σ = 0.02;
- jet mass: propagate from the smeared four-vector.

The flat 20% smearing currently in `configs/default.yaml` remains available
as `pt_smear_frac` for comparison runs only.

---

## 4. Metrics (frozen)

All metrics are computed on the test split, on reconstructable signal
events unless stated, with the same smearing realisation for model and
baseline. Report a binomial uncertainty √(p(1−p)/N) for every rate.

### 4.1 Assignment

| name | definition |
|---|---|
| `acc` | fraction of events whose argmax hypothesis equals the truth hypothesis |
| `acc_isr` | fraction whose predicted extra-jet set equals truth |
| `acc_grp` | fraction whose 3+3 grouping is right given the true extra-jet set |
| `acc_by_nj` | `acc` per jet multiplicity (6, 7, 8, 9+) |
| `acc_classical` | `acc` of min \|m1−m2\| over valid hypotheses on the same events |
| `gain` | `acc` − `acc_classical` with combined uncertainty |

Valid hypotheses exclude any that place a padded or masked slot inside a
triplet.

### 4.2 Physics

Signal region (SR): mass asymmetry A = \|m1−m2\|/(m1+m2) < 0.1, and, once a
topology head exists, topology score above the threshold that gives 50%
signal efficiency on the 1000 GeV validation sample. The threshold is
chosen once per experiment on validation and frozen for test.

| name | definition |
|---|---|
| `eff_S(M)` | fraction of all generated signal events at mass M that land in the SR with \|m_avg − M\| < 0.1 M |
| `B(M)` | weighted QCD count in the SR with \|m_avg − M\| < 0.1 M |
| `Z(M)` | eff_S(M) / √B(M), reported relative to the same quantity for the classical solver |
| `sculpt` | QCD m_avg histogram passing the topology cut divided by the histogram failing it, fitted to a constant; report χ²/ndf and the largest bin deviation in σ |
| `disco` | distance correlation between topology score and m_avg on test QCD |
| `mass_loo(M)` | eff_S(M) when M was excluded from training, divided by eff_S(M) when included |

`sculpt` and `disco` are guardrail metrics: an experiment that improves `Z`
but worsens `sculpt` beyond χ²/ndf = 2 is `reverted`, not `kept`.

### 4.3 Standard plots per experiment

All PDF, all Tufte, all in `experiments/<id>/plots/`:

1. `acc` and `acc_classical` versus epoch, validation, baseline as a thin
   rule with a direct label.
2. `acc_by_nj` as a dot plot with error bars, model and baseline side by
   side.
3. m_avg distributions on test: signal correct, signal wrong, QCD; one
   panel per region (all, SR).
4. When a topology head exists: signal efficiency versus QCD rejection, and
   the `sculpt` ratio plot with its fitted constant.
5. When a mass grid exists: `eff_S(M)` and `Z(M)` versus M, model and
   classical.

---

## 5. The loop, per experiment

1. **Select.** Take the first `todo` item in `experiments/QUEUE.md` whose
   dependencies are `kept` or `done`. Append a row to
   `experiments/ledger.csv` with `status=running`, the hypothesis, the
   primary metric and the success criterion copied from the queue.
2. **Branch and implement.** `git checkout -b exp/<id>-<slug>` from the
   loop branch. Make the one change. Write or update a config under
   `configs/exp/<id>.yaml`. Run the two-minute smoke test:
   `python -m src.train --config configs/exp/<id>.yaml --data <one small file>`
   with `num_epochs: 1`. Commit. Push.
3. **Data.** Check `data/MANIFEST.md` for a sample that already satisfies
   the experiment. Reuse it. Otherwise generate per section 3, validate,
   record, commit the manifest.
4. **Train.** Under `nohup` with the log at `experiments/<id>/train.log`.
   Poll the log rather than blocking. Kill and mark `failed` on NaN loss,
   on exceeding the budget cap, or on validation `acc` below
   `acc_classical` after 30% of the cap.
5. **Evaluate.** `python -m src.evaluate` on the frozen test split, writing
   metrics to `experiments/<id>/metrics.json`. If the primary metric moved
   by less than three single-run uncertainties, run two more seeds and
   report mean and spread.
6. **Plot.** The standard set in section 4.3.
7. **Report.** Fill `experiments/<id>/report.md` from the template. Set the
   ledger row to `kept`, `reverted`, `inconclusive` or `failed`, with the
   numbers.
8. **Decide.** `kept` if the primary metric improved beyond noise and no
   guardrail metric regressed. Merge the experiment branch into the loop
   branch. Otherwise leave the branch in place, do not merge, and record why.
9. **Requeue.** Mark the item done in `QUEUE.md`. Add follow-ups the result
   suggests, with the same fields, below the existing items. Re-order only
   if a result makes a later item moot.
10. **Repeat** until the budget is spent, the deadline is near, or the
    queue is empty. Then write `experiments/REPORT_<date>.md`: ledger
    summary table, best configuration and its numbers against the
    classical solver, the five most informative plots, open questions,
    and the recommended next queue. Push. Tell the user in a short message
    with the report path.

---

## 6. Training conventions

- Seeds: 1, 2, 3. Set for torch, numpy and the loader.
- Optimiser, schedule and batch size from `configs/default.yaml` unless the
  experiment is about them.
- Log every epoch to `logs/<id>_training_log.csv` with the classical
  baseline on the validation split as an extra column.
- Save only `checkpoints/<id>_best.pt`.
- Default caps: 2 GPU-hours per training run, 6 GPU-hours per experiment
  including seeds. `QUEUE.md` may override per item.
- MPS specifics: the device is selected automatically; everything stays
  float32 because MPS has no float64; `num_workers` stays 0 because the
  loader is in memory; `pin_memory` is irrelevant. Unified memory allows
  large batches, but the GroupTransformer's 140 mini-transformer passes
  per event set the epoch time, so keep batch 256 unless the smoke test
  argues otherwise.
- Budgets are wall-clock hours on the Mac's GPU. Set each run's cap from
  the smoke test's seconds per epoch times the planned epochs, and write
  the cap into the ledger row before the run starts.
- If a checkpoint gives validation accuracy on MPS that differs from CPU
  by more than the binomial error, treat it as a bug: evaluate on CPU for
  that experiment and open a follow-up queue item.

---

## 7. Reporting format

`experiments/ledger.csv` columns:

```
id,slug,status,branch,commit,data_manifest_ids,primary_metric,baseline,result,uncertainty,seeds,gpu_hours,decision_reason,report_path,started,finished
```

`status` is one of `running`, `kept`, `reverted`, `inconclusive`, `failed`,
`superseded`.

The final loop report is the only message the user must read. Everything
else is in the repo.

---

## 8. Guardrails: stop and ask the user

Stop the loop and ask when any of these holds. Finish and push everything
that does not depend on the answer first.

- A change would alter the signal-region definition or a metric definition.
- The budget will be exhausted before the current experiment can finish
  and the partial result would be uninformative.
- Three seeds disagree on the sign of the primary metric change.
- A producer fix requires a physics choice not covered here (a different
  decay mode, a different jet radius, a different collider energy).
- Anything destructive outside `checkpoints/`, `logs/`, `plots/`,
  `data/raw/` and intermediate generation files.
- Access to a repository, image or dataset the session does not have.

Never: skip or weaken a validation check to make a file usable; train on
the test split; change a frozen definition mid-loop; push to `main`; open
a pull request unless the handoff asks for one.

---

## 9. Established facts, do not re-derive

From the audit of 2026-09-23 on the two files in `data/`. Re-measure only
if the data or the code path changes.

- Both files are parton level, no shower, no detector, exactly 1 TeV
  gluinos with \|m1−m2\| median under 1 GeV at truth.
- `gogoj_offshell_10k.h5`: in every 7-parton event the extra gluon is at
  file index 0 and the stored TARGETS are wrong. The six non-gluon partons
  in file order, split [1,2,3] and [4,5,6], reconstruct both gluinos.
- `gluino_rpv_1tev_uds_10000evt_20260225_164524.h5`: six partons per
  event, slot 6 is zero-padded. 60 of the 70 hypotheses put the pad inside a
  triplet. The argmin \|m1−m2\| label picks one of those in 8% of events.
- Classical solver, 20% flat pT smearing, 1 TeV file: 7.6% accuracy with
  pad-in-triplet hypotheses allowed, 31% with them masked.
- Unsmeared argmin label agrees with truth 89.6% (pad allowed) and 97.4%
  (pad masked).
- Current default model has 7.1M parameters; the README describes a 1.0M
  one. Forward pass of 64 events takes 0.6 s on CPU.
- The padded slot produces a pT ratio of order 10^7 in its triplet and an
  attention bias of 16.5 against 0.45 for real pairs.
- `configs/default.yaml` enables three signal-side QCD proxy losses with no
  QCD sample present; the background-rejection loss never fires.
- Phase 2 distillation uses a teacher whose probabilities span a ratio of
  1.28 across 70 classes; it acts as a push toward uniform for 20 epochs.
- The column-based truth fallback reads columns 4 and 5 for `parent_pdg`
  and `is_signal`; the files have them at 5 and 6.
