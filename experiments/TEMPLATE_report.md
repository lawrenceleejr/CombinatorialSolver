# <id> <slug>

- status: <kept | reverted | inconclusive | failed>
- branch / commit: `exp/<id>-<slug>` / `<hash>`
- data: manifest ids `<ids>`; tier <k>; split salt as in `data/splits/`
- budget: <used> of <cap> GPU-hours
- seeds: <list>

## Hypothesis

One sentence, copied from the queue.

## Change

What changed, in one paragraph. Point to the diff rather than pasting it.

## Result

| metric | baseline | result | uncertainty |
|---|---|---|---|
| primary | | | |
| acc_classical | | | |
| guardrails | | | |

Per-seed values if more than one seed ran.

## Plots

Paths under `plots/`, one line each on what the plot shows.

## Decision

Kept or reverted, and the one reason. What this result changes about the
queue.

## Notes

Anything surprising, anything that failed and was retried, anything the next
session should know.
