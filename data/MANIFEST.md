# Data manifest

One entry per generated or imported file. Append only. A file that is not
listed here is not used for training or evaluation.

| id | path | tier | producer commit | config | mass (GeV) | seed | events | validation | notes |
|---|---|---|---|---|---|---|---|---|---|
| D0 | `data/gluino_rpv_1tev_uds_10000evt_20260225_164524.h5` | 0 | unknown, pre-fix | `gluino_rpv_1tev_uds.yaml`, `--shower off` | 1000 | unknown | 10000 | 6 partons per event, truth grouping correct, m1 = m2 = 1000 | zero-padded 7th slot; see DESIGN_LOOP section 9 |
| D1 | `data/gogoj_offshell_10k.h5` | 0 | unknown, pre-fix | `p p > go go` plus 1 extra parton, `--shower off` | 1000 | unknown | 10000 | 60% of events have 7 partons; stored TARGETS wrong for those | gluon at index 0; correct grouping is non-gluon partons [1,2,3],[4,5,6] |

Validation column records the section 3.4 checks. Generation rate, measured
from the smoke run, goes in the notes of the first file generated in a loop.
