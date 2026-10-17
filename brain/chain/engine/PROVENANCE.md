# Provenance

Copied unchanged from github.com/Nikku03/cell, `rem/atlas/`, branch `claude/vectorize-gex-propensity-zp09w8`,
commit `762f2ec2b421fd3bd17cafe708344d217580508e`:

- `chainexact.py`: the bump-node dendritic tree (24 units, 16 branches merging in groups of 4) and the chain task.
- `chainsnap.py`: the loop that writes down one item per step (the snap).
- `biomemory.py`: the fast-fading synaptic memory for facts shown once (not used by `brain/chain`, which shows
  the links every step).

Their own results and gates are in that repository's `rem/atlas/RESULTS_*.txt`. Running a file's `main()` writes
results next to it; `brain/chain` only imports the classes and helpers.
