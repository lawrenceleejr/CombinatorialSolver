#!/bin/bash
# State of the F5 jobs: condor status and every epoch line so far (log in runs/ once the job has
# ended, plus the live stdout through condor_tail while it runs).  ./status.sh [all] (all = every epoch)
D=/scratch/llee/pairnet/audit/F5_train
while read n c; do
  st=$(condor_q $c -af JobStatus NumJobStarts 2>/dev/null); [ -z "$st" ] && st="left queue"
  echo "== $n $c [JobStatus NumJobStarts: $st]"
  { cat $D/runs/out_$n/$n.train.log 2>/dev/null; cat /var/lib/condor/spool/$((c % 10000))/0/cluster$c.proc0.subproc0/out/$n.train.log 2>/dev/null; [ "${st%% *}" = 2 ] && timeout 45 condor_tail -maxbytes 400000 $c 2>/dev/null; } |
    grep -a '^{"n_sig' | /scratch/llee/envs/pairnet/bin/python -c '
import sys, json
seen = {}
for l in sys.stdin:
    try: r = json.loads(l)
    except Exception: continue
    seen[r["epoch"]] = r
eps = sorted(seen) if len(sys.argv) > 1 else sorted(seen)[-1:]
for e in eps:
    r = seen[e]; p = r.get("sig_at_psig_bkg") or {}
    print("   ep %2d loss %.4f val %.4f train_sub %.4f gap %+.4f | auc %.4f rej50 %.1f bkgA01 %.4f sigwin %.4f psig@0.5%% %s (train_sub %s) %.0fs" % (
        e, r["loss"], r["val_loss"], r["train_eval_loss"], r["val_loss"] - r["train_eval_loss"], r["auc"], r["bkg_rej_at_50"],
        r["bkg_pass_asym"], r["sig_pass_asym_in_window"], p.get("0.005"), (r["train_sub"].get("sig_at_psig_bkg") or {}).get("0.005"), r["time"]))
' $1
done < $D/clusters.txt
