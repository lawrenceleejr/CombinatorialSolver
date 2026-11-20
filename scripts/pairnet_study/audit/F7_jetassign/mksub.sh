#!/bin/bash
# mksub.sh <name> <extra args...> : OSPool GPU submit file for one JetAssign training
N=$1; shift
ARGS=$(sed "s/\([^ ]*[*[][^ ]*\)/'\1'/g" args_base.txt | tr -d '\n')
cat > sub/$N.sub <<S
executable     = /scratch/llee/pairnet/audit/F7_jetassign/train_ja.sh
arguments      = "10800 $ARGS --threads 2 --out out/$N $*"
transfer_input_files = osdf:///ospool/uc-shared/project/futurecolliders/llee/pairnet/train_env_v1.tar.gz, osdf:///ospool/uc-shared/project/futurecolliders/llee/pairnet/audit_f7/data_f7_cache.tar, /scratch/llee/pairnet/audit/F7_jetassign/repo_f7.tar.gz
transfer_output_files = out
transfer_output_remaps = "out = /scratch/llee/pairnet/audit/F7_jetassign/runs/out_$N"
checkpoint_exit_code = 85
transfer_checkpoint_files = out
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
output         = /scratch/llee/pairnet/audit/F7_jetassign/logs/$N.out
error          = /scratch/llee/pairnet/audit/F7_jetassign/logs/$N.err
log            = /scratch/llee/pairnet/audit/F7_jetassign/logs/$N.log
+JobDurationCategory = "Long"
+JobBatchName = "f7_$N"
request_gpus   = 1
max_retries    = 3
success_exit_code = 0
gpus_minimum_memory = 7000
request_cpus   = 2
request_memory = 20GB
request_disk   = 20GB
queue
S
echo sub/$N.sub
