#!/bin/bash
# r15: r14A recipe + enriched hard-negative QCD (in-job r14A filter: P >= 0.8 or boosted Delta, A < 0.1) in training,
# epoch chosen at the low working point (signal in window at 0.5% of A<0.1 QCD by P(signal)). Two variants: bkg weight 3, 6.
set -e
P=/scratch/llee/pairnet; OSDF=/ospool/uc-shared/project/futurecolliders/llee/pairnet; T=$OSDF/tail
S=$P/stage_r15; rm -rf $S; mkdir -p $S/tail
for f in $(ls $T | grep -E '^qcdvlmT_c([0-9]|[1-9][0-9]|[1-8][0-9][0-9])_tail\.h5$|^qcdhtT_c([0-9]|[1-9][0-9]|[1-4][0-9][0-9])_tail\.h5$'); do ln -s $T/$f $S/tail/$f; done
echo "tail training files: $(ls $S/tail | wc -l)"
tar -chf $OSDF/data_tail_v1.tar -C $S tail
tar --exclude=.git -czf $OSDF/repo_r15.tar.gz -C /home/llee/CombinatorialSolver .
ls -la $OSDF/data_tail_v1.tar $OSDF/repo_r15.tar.gz
cd $P/sweep
SIG="'prod/sig_*_c*[0-7].h5' 'lm/siglm_m*_incl_c[0-9].h5' 'lm/siglm_m*_incl_c1[0-5].h5' 'lm/siglm_m*_boost_c[0-9].h5' 'lm/siglm_m*_boost_c1[0-5].h5'"
VSIG="'prod/sig_*_c*8.h5' 'lm/siglm_m*_incl_c1[67].h5' 'lm/siglm_m*_boost_c1[67].h5'"
TSIG="'prod/sig_*_c*9.h5' 'lm/siglm_m*_incl_c1[89].h5' 'lm/siglm_m*_boost_c1[89].h5'"
BKG="'prod/qcd_c*[0-7].h5' 'prod_ht/qcdht_c[0-9].h5' 'prod_ht/qcdht_c[0-9][0-9].h5' 'prod_ht/qcdht_c[0-9][0-9][0-9].h5' 'prod_ht/qcdht_c1[0-3][0-9][0-9].h5' 'lm/qcdvlm_c[0-9].h5' 'lm/qcdvlm_c[0-9][0-9].h5' 'lm/qcdvlm_c[0-9][0-9][0-9].h5' 'tail/qcdvlmT_c*_tail.h5' 'tail/qcdhtT_c*_tail.h5'"
VBKG="'prod/qcd_c*8.h5' 'prod_ht/qcdht_c1[4-6][0-9][0-9].h5' 'lm/qcdvlm_c10[0-4][0-9].h5'"
TBKG="'prod/qcd_c*9.h5' 'prod_ht/qcdht_c1[7-9][0-9][0-9].h5' 'lm/qcdvlm_c10[5-9][0-9].h5'"
for V in A:3 B:6; do IFS=: read tag bw <<< "$V"; N=r15${tag}_small_12j_win5_tailqcd_bw$bw
cat > $N.sub <<EOS
executable     = /scratch/llee/pairnet/train_auto.sh
arguments      = "5400 --gated --proxy --gate-interp --mass-head --sig $SIG --bkg $BKG --val-sig $VSIG --val-bkg $VBKG --test-sig $TSIG --test-bkg $TBKG --device auto --cache-dir cache --threads 2 --out out/$N --n-jets 12 --n-assign 7 --epochs 12 --bs 1024 --d-model 128 --layers 4 --lr 2e-3 --bkg-weight $bw --select psig_at_bkg --select-bkg-pass 0.005 --w-win 5 --w-qcd-pass 5"
transfer_input_files = osdf://$OSDF/train_env_v1.tar.gz, osdf://$OSDF/data_r3_v1.tar.gz, osdf://$OSDF/data_ht_v1.tar, osdf://$OSDF/data_lm_v1.tar, $OSDF/data_tail_v1.tar, $OSDF/repo_r15.tar.gz
transfer_output_files = out
transfer_output_remaps = "out = $OSDF/sweep_r15/out_$N"
checkpoint_exit_code = 85
transfer_checkpoint_files = out, cache
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
output         = /scratch/llee/pairnet/sweep/logs/$N.out
error          = /scratch/llee/pairnet/sweep/logs/$N.err
log            = /scratch/llee/pairnet/sweep/logs/$N.log
+JobDurationCategory = "Long"
request_gpus   = 1
max_retries    = 3
success_exit_code = 0
requirements   = !regexp("qa-rtx6k-020", Machine)
gpus_minimum_memory = 7000
request_cpus   = 2
request_memory = 24GB
request_disk   = 40GB
queue
EOS
condor_submit $N.sub | tail -1
done
