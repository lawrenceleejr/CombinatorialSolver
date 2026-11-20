#!/bin/bash
# Generate the HTCondor submit files of the F5 audit batch (one job per training) into sub/.
# Data: the r15 array cache already on OSDF (r14A training set = its first 17,396,294 events; the rest
# is the r15 training tail QCD) + the never-trained tail evaluation arrays (data_f5_tail_v1.tar).
set -e
F=/scratch/llee/pairnet/audit/F5_train; OSDF=/ospool/uc-shared/project/futurecolliders/llee/pairnet
N14=17396294
BASE="--gated --proxy --gate-interp --mass-head --train-npz cache/train_a57f78a76e60.npz --val-npz cache/val_8ddd403e85f3.npz --test-npz cache/test_2da66480a069.npz --extra-eval tail=f5/tail_eval.npz --eval-mass-variants mup mdn msm m0 --save-logits --device auto --threads 2 --n-jets 12 --n-assign 7 --bs 1024 --bkg-weight 3 --select win_a01 --w-win 5 --w-qcd-pass 5"
SMALL="--d-model 128 --layers 4 --lr 2e-3"
mk() {  # name  budget(s)  gpu-mem  gpu-capability  args...
  local N=$1 B=$2 GM=$3 GC=$4; shift 4
  cat > $F/sub/$N.sub <<EOS
executable     = $F/train_f5.sh
arguments      = "$((B + 3600)) $BASE --time-budget $B --out out/$N $*"
transfer_input_files = osdf://$OSDF/train_env_v1.tar.gz, osdf://$OSDF/data_cache_r15.tar, osdf://$OSDF/audit_f5/data_f5_tail_v1.tar, $F/stage/repo_f5.tar.gz
transfer_output_files = out
transfer_output_remaps = "out = $F/runs/out_$N"
checkpoint_exit_code = 85
transfer_checkpoint_files = out
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
output         = $F/logs/$N.out
error          = $F/logs/$N.err
log            = $F/logs/$N.log
+JobDurationCategory = "Long"
request_gpus   = 1
max_retries    = 3
success_exit_code = 0
requirements   = !regexp("qa-rtx6k-020", Machine)
gpus_minimum_memory = $GM
$( [ $GC != 0 ] && echo "gpus_minimum_capability = $GC" )
request_cpus   = 2
request_memory = 24GB
request_disk   = 30GB
queue
EOS
}
R14="--train-n $N14"
for s in 1 2 3; do mk f5_s$s 7200 11000 0 $R14 $SMALL --epochs 12 --seed $s; done      # 1. run-to-run noise, ensemble members
mk f5_e30      7200 11000 0 $R14 $SMALL --epochs 30                                      # 2. longer
mk f5_drop0    7200 11000 0 $R14 $SMALL --epochs 12 --dropout 0                          #    no dropout (no train/val gap in r14A)
mk f5_big6     10800 15000 8.0 $R14 --d-model 256 --layers 6 --lr 1e-3 --epochs 20       # 3. bigger
mk f5_big8     10800 15000 8.0 $R14 --d-model 256 --layers 8 --lr 1e-3 --epochs 20
mk f5_big6tail 10800 15000 8.0 --d-model 256 --layers 6 --lr 1e-3 --epochs 20            #    + r15 training tail QCD
mk f5_wp       7200 11000 0 $R14 $SMALL --epochs 12 --gate-wp-qcd 5 --gate-wp-sig-fail 0.2   # 4. loss at the working point
mk f5_phys     7200 11000 0 $R14 $SMALL --epochs 12 --qcd-phys-weight 10
mk f5_focal    7200 11000 0 $R14 $SMALL --epochs 12 --gate-focal 2
mk f5_maug     7200 11000 0 $R14 $SMALL --epochs 12 --mass-aug 0.1,0.1,0.2               # 5. jet-mass robustness
mk f5_m0       7200 11000 0 $R14 $SMALL --epochs 12 --massless
ls $F/sub/*.sub | wc -l
