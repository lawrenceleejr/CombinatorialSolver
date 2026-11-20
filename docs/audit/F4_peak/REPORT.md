# F4 — peak ceiling with ancestry truth (written by main from the agent's final message; full tables in out/tables.md)
Truth: generate_pythia.py --ancestry (commit 2cc7ce1, branch audit/f4-ancestry, worktree wt/) labels each final parton g1/g2/other,
ghost-associated to R=0.4 jets (INPUTS/Source/anc; ANC/* for jets >20 GeV). Samples: samples/sigan_m{400,700,1000,1500,2000}_c00..11.h5
(120k/mass, keep failing events: cut EventVars/pass_sel).

% of all events in [0.85,1.1]M / [0.92,1.04]M (groupings chosen by ancestry, not mass)
scheme                             400        700        1000       1500       2000
r14A as run                        17.2/9.2   28.8/16.2  34.7/19.9  45.0/26.9  59.3/36.3
best 3+3 in leading 7              24.8/14.5  37.3/23.2  43.8/28.1  49.4/32.7  52.4/35.6
majority, free sizes, leading 10   43.4/26.2  62.5/44.5  70.1/55.3  74.8/64.5  76.5/68.9
pT>20, leading 12                  53.4/34.5  68.7/52.9  73.9/62.1  77.0/69.7  77.9/72.8
Peak 68% half-width at 1 TeV: r14A 0.205; 3+3 truth 0.066; free sizes 0.041. r14A in-window at 1 TeV: 9.0% correct vs 25.7% wrong groupings.
Losses: jet pT>30 is the biggest at low mass (34% of 400 GeV events have a decay quark <30 GeV); forcing 3+3 costs 15–22 points.
QCD cost (A<0.1 & wide window, "any grouping"): sizes 2–5 in 10 is ×1.6–2.3 vs 3+3 in 7; min-asymmetry rates drop. Net S/√B ×1.2 to large.
Simple ΔR attach rule fails (signal 34.7→31.5%, QCD 2.79→3.58% at 1 TeV).
Recommendation: per-jet g1/g2/other head over ≤12 jets (pT>20), loss = min over label swap, soft-assigned group 4-vectors; keep gate and
mass head. Projected narrow-window signal 20→39% (1 TeV), 27→47% (1.5 TeV), peak ~2× narrower; QCD factor must be measured.
