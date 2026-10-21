# Runbook: produce the datasets on a cluster and train on a GPU

Written for an agent with shell access to a batch cluster (Slurm or HTCondor)
that has CPU nodes and at least one GPU node. Follow the steps in order; each
ends with a **checkpoint**. Do not continue past a failed checkpoint: fix the
cause first (common causes are listed at the end).

Everything lives on branch `claude/pairnet-review` of
`https://github.com/lawrenceleejr/CombinatorialSolver`.

Budget: about 40 CPU core-hours for the datasets (under an hour on 64 cores),
~2 GB of disk, and roughly 1–2 GPU-hours for training.

## 0. Variables used below

Set these once per shell (edit paths for the cluster):

```bash
export REPO=$HOME/CombinatorialSolver          # code checkout (shared filesystem)
export DATA=/path/to/shared/scratch/pairnet    # large shared disk, visible from all nodes
export ENVDIR=$HOME/envs/pairnet               # python environment (shared filesystem)
```

`$REPO`, `$DATA` and `$ENVDIR` must be on a filesystem that batch nodes can
see. If worker nodes cannot see home, put all three on the shared scratch.

## 1. Get the code

```bash
git clone https://github.com/lawrenceleejr/CombinatorialSolver $REPO
cd $REPO && git checkout claude/pairnet-review && git rev-parse --short HEAD
```

Record the commit hash; put it in the names of what you produce.

## 2. Build the environment (once, on a login or interactive node)

Python 3.10+ is needed. Prefer conda/mamba if the cluster has it (prebuilt
Pythia and FastJet):

```bash
# with conda / mamba
conda create -y -p $ENVDIR -c conda-forge python=3.11 pythia8 fastjet
conda activate $ENVDIR
pip install -r $REPO/requirements.txt onnxruntime

# without conda: plain venv, Pythia is compiled from source (~10-20 min)
python3 -m venv $ENVDIR && source $ENVDIR/bin/activate
pip install -r $REPO/requirements.txt onnxruntime
USE_CONDA=0 JOBS=8 bash $REPO/scripts/install_generators.sh
```

On the GPU node, PyTorch must be a CUDA build. If `pip install torch` gave a
CPU-only wheel, reinstall from the matching CUDA index
(`pip install torch --index-url https://download.pytorch.org/whl/cu121`, with
the CUDA version the node's driver supports).

**Checkpoint 2:**

```bash
python -c "import pythia8mc" 2>/dev/null || python -c "import pythia8"   # either import works
python -c "import fastjet, h5py, torch, onnx, onnxruntime; print('ok')"
```

## 3. Smoke test (one minute, interactive)

```bash
cd $REPO
python scripts/generate_pythia.py --process signal --mass 1000 --n 200 --seed 1 --truth-quarks before_fsr --out $DATA/smoke/sig.h5
python scripts/generate_pythia.py --process qcd --n 100 --min-jets 5 --seed 1 --out $DATA/smoke/qcd.h5
python - <<'PY'
import h5py, os
d = os.environ["DATA"] + "/smoke/"
s = h5py.File(d + "sig.h5"); q = h5py.File(d + "qcd.h5")
print("signal", s["INPUTS/Source/pt"].shape, "truth keys", list(s["truth"]), "matched with", s.attrs.get("truth_quarks"))
print("qcd", q["INPUTS/Source/pt"].shape)
PY
```

**Checkpoint 3:** both files are written, signal shows
`truth keys ['jet_index', 'quarks', 'quarks_after_fsr', 'quarks_before_fsr']`
and `matched with before_fsr`. Signal runs at roughly 50 events/s per core;
QCD at roughly 10 kept events/s.

## 4. Produce the datasets on CPU nodes

Write the job list (one generator command per line, each makes one file of
10k events with its own fixed seed):

```bash
cd $REPO
python scripts/produce_samples.py --outdir $DATA/prod \
    --masses 400:2400:100 --sig-events 100000 --qcd-events 1000000 \
    --joblist $DATA/jobs.txt
wc -l $DATA/jobs.txt        # expect 21 masses x 10 + 100 QCD = 310 lines
```

What this produces:

- signal `sig_m{M}_c{i}.h5`, 21 masses from 400 to 2400 GeV, 100k events each;
- QCD `qcd_c{i}.h5`, 1M events with at least 5 jets (the 5-jet events serve your
  hemisphere-mixing control region; training uses the 6+ jet ones);
- the file index `i` sets the split: `i % 10` in 0–7 is training, 8 is
  validation, 9 is test;
- jets are matched to the decay quarks before their own shower; both truth
  definitions are stored.

The list is resumable: rerunning `produce_samples.py` writes only the
commands whose output file does not exist yet. Files are written to a `.tmp`
name and renamed at the end, so a killed job never leaves a file that looks
finished.

### Slurm

```bash
mkdir -p $DATA/logs
cat > $DATA/gen.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=pairnet-gen
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --output=logs/gen_%A_%a.out
source $ENVDIR/bin/activate 2>/dev/null || conda activate $ENVDIR
cd $REPO
eval "$(sed -n "${SLURM_ARRAY_TASK_ID}p" $DATA/jobs.txt)"
EOF
cd $DATA && sbatch --export=ALL --array=1-$(wc -l < jobs.txt)%200 gen.sbatch
```

Signal files take about 5 minutes each, QCD files about 20–30 minutes each.
`%200` caps concurrent jobs; adjust to the cluster's policy.

### HTCondor

Create the wrapper `$DATA/gen_one.sh` (it runs line `N+1` of `jobs.txt`,
exactly like the Slurm script), replacing the three paths with the values of
`$ENVDIR`, `$REPO` and `$DATA`:

```bash
#!/bin/bash
source /path/to/envs/pairnet/bin/activate 2>/dev/null || conda activate /path/to/envs/pairnet
cd /path/to/CombinatorialSolver
eval "$(sed -n "$(( $1 + 1 ))p" /path/to/shared/scratch/pairnet/jobs.txt)"
```

Then the submit file `$DATA/gen.sub` (again with real paths, and `N` = the
number of lines in `jobs.txt`):

```
executable     = /path/to/shared/scratch/pairnet/gen_one.sh
arguments      = $(Process)
output         = /path/to/shared/scratch/pairnet/logs/gen_$(Process).out
error          = /path/to/shared/scratch/pairnet/logs/gen_$(Process).err
log            = /path/to/shared/scratch/pairnet/logs/gen.log
request_cpus   = 1
request_memory = 2GB
queue N
```

```bash
chmod +x $DATA/gen_one.sh && mkdir -p $DATA/logs && condor_submit $DATA/gen.sub
```

### No batch system (one big node)

```bash
python scripts/produce_samples.py --outdir $DATA/prod --masses 400:2400:100 \
    --sig-events 100000 --qcd-events 1000000 --cores $(nproc)
```

**Checkpoint 4** (after all jobs finish):

```bash
ls $DATA/prod/sig_*.h5 | wc -l     # 210
ls $DATA/prod/qcd_*.h5 | wc -l     # 100
ls $DATA/prod/*.tmp 2>/dev/null    # nothing
# resubmit only what is missing:
python scripts/produce_samples.py --outdir $DATA/prod --masses 400:2400:100 \
    --sig-events 100000 --qcd-events 1000000 --joblist $DATA/jobs_retry.txt
wc -l $DATA/jobs_retry.txt         # 0 when complete
```

Then check one file per type opens and has events:

```bash
python - <<'PY'
import h5py, glob, os
for pat in ("sig_m1000_c0.h5", "qcd_c0.h5"):
    f = h5py.File(os.path.join(os.environ["DATA"], "prod", pat))
    print(pat, f["INPUTS/Source/pt"].shape, dict(f.attrs).get("truth_quarks"))
PY
```

## 5. Train on the GPU

This reproduces the recommended model (`models/pairnet_gated_7jet_mass.onnx`):
7 jets, gated, the gate sees the pairing summary, plus the mass head. With 100k
signal events per mass instead of 10k, raise the batch size and use the
validation and test files explicitly.

```bash
cat > $DATA/train.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=pairnet-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.out
source $ENVDIR/bin/activate 2>/dev/null || conda activate $ENVDIR
cd $REPO
nvidia-smi
P=$DATA/prod
python scripts/train_pairnet.py --gated --proxy --gate-interp --mass-head --n-jets 7 \
    --sig "$P/sig_*_c*[0-7].h5"  --bkg "$P/qcd_c*[0-7].h5" \
    --val-sig "$P/sig_*_c*8.h5"  --val-bkg "$P/qcd_c*8.h5" \
    --test-sig "$P/sig_*_c*9.h5" --test-bkg "$P/qcd_c*9.h5" \
    --bkg-weight 3 --epochs 40 --bs 1024 --lr 2e-3 --amp --device auto \
    --select combined --cache-dir $DATA/cache --out $DATA/runs/g7_mass
EOF
cd $DATA && sbatch --export=ALL train.sbatch
```

Notes:

- Keep the glob patterns quoted; the script expands them itself.
- The split is set by the last digit of the file index (`c*8` = validation, `c*9` = test), so the
  patterns must be `c*[0-7]`, `c*8`, `c*9`: QCD files run from `qcd_c0` to `qcd_c99`, and `qcd_c[0-7]`
  would pick only 8 of the 80 training files.
- `--bkg-weight 3` rebalances QCD against the much larger signal set.
- `--cache-dir` stores the parsed arrays after the first read. Parsing ~3M
  events from h5 takes several minutes; later runs start in seconds. Delete
  the cache if the input files change.
- `--resume` continues from `$DATA/runs/g7_mass/last.pt` (written every
  epoch) if the job hits its time limit; resubmit with the same arguments
  plus `--resume`.
- `--amp` (bfloat16) needs an Ampere or newer GPU; drop it on older cards.
- If memory is tight, lower `--bs` to 512 and `--lr` to 1.4e-3.

**Checkpoint 5:** the log prints one JSON line per epoch. On the original
small testbed this configuration reached, on validation, `auc` ≈ 0.965,
`eff_on_reconstructable` ≈ 0.22–0.30 (exact pairing over events whose truth
is in the leading 7 jets, all masses pooled; the higher end with the
before-FSR truth these files use), and `m_est_rel_halfwidth68` ≈ 0.18. With 10× the data
expect these or better. The run ends with a `TEST {...}` line and writes
`model.pt`, `last.pt`, `metrics.json`, `eval.npz`. Red flags: `auc` below
0.90 after 5 epochs, `loss` NaN, or `m_est_rel_halfwidth68` above 0.25.

## 6. Export, validate, evaluate

```bash
cd $REPO
H=$(git rev-parse --short HEAD)
python scripts/export_pairnet_onnx.py --model $DATA/runs/g7_mass/model.pt \
    --out $DATA/models/pairnet_gated_7jet_mass_$H.onnx
```

`validate_onnx.py` and the evaluation scripts default to the testbed paths in
`data/pythia`; point them at the new test files by linking one test file per
mass into those names, or run the equivalent checks directly:

```bash
mkdir -p data/pythia
ln -sf $DATA/prod/sig_m1000_c9.h5 data/pythia/sig_m1000_s2.h5
ln -sf $DATA/prod/qcd_c9.h5      data/pythia/qcd_s8.h5
python scripts/validate_onnx.py --onnx $DATA/models/pairnet_gated_7jet_mass_$H.onnx \
    --model $DATA/runs/g7_mass/model.pt
```

**Checkpoint 6:** argmax agreement 1.0000 for both `logits` and
`signal_logits`, `max|d p_signal|` below 1e-3, HT-normalised input check
`True`, and a `m_est` peak narrower than m_avg.

Pairing efficiency per mass, scored on before-FSR truth (link the test files
as `data/pythia_bf/sig_m{M}_s2.h5` first):

```bash
mkdir -p data/pythia_bf
for M in 500 1000 1500 2000; do ln -sf $DATA/prod/sig_m${M}_c9.h5 data/pythia_bf/sig_m${M}_s2.h5; done
python scripts/truth_definition.py --masses 500 1000 1500 2000 \
    --models new=$DATA/runs/g7_mass/model.pt --out $DATA/truth_definition.json
```

Compare with the testbed model: 27–32% of available events at 500–2000 GeV.

Working points for the analysis, after running the search study on the new
test files (`scripts/search_study.py --sig "data/pythia/sig_*_s2.h5"` with the
same kind of links), come from `scripts/working_points.py`.

## 7. What to hand back

- the ONNX file and its `.json` spec, named with the commit hash;
- `$DATA/runs/g7_mass/metrics.json` and the training log;
- the output of checkpoints 5 and 6 and the pairing table;
- the exact `train_pairnet.py` command used.

## Optional runs, once the main one works

- **Shower-robust training** (reduces the 11–17% signal-efficiency dependence
  on shower α_s): produce extra samples with `generate_pythia.py --alphas-fsr
  0.150 --alphas-isr 0.150` and `0.123` (both signal and QCD; easiest is to
  write a second job list by editing the commands in `jobs.txt` and changing
  `--outdir`/output names and seeds), then train on the union. Check with
  `scripts/robustness.py`.
- **6-jet variant** for hemisphere mixing: same command with `--n-jets 6`.
- **Massless test**: add `--massless` (costs about ×1.7 signal at fixed QCD on
  the testbed, but removes the shower dependence).

## Common failures

| symptom | cause and fix |
|---|---|
| `ImportError: pythia8mc` / `pythia8` on worker nodes | environment not visible or not activated on the node; put `$ENVDIR` on a shared filesystem and activate it inside the job script |
| `Pythia init failed` | the SLHA temp directory is not writable; set `TMPDIR` to a node-local writable directory in the job script |
| many `.tmp` files left | jobs hit the time limit; raise `--time` (QCD files need up to ~30 min) and rerun the retry list |
| training uses the CPU | CPU-only torch wheel; reinstall the CUDA build and confirm `torch.cuda.is_available()` |
| out of GPU memory | lower `--bs`; the dataset itself is ~1 GB on the device |
| `no signal events` / empty split | glob patterns not quoted or wrong `$DATA` path; print `ls "$P"/sig_*_c8.h5` |
