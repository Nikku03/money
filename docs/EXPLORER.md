# The scan explorer

A private web page that sets every stock of the latest scan side by side, built on
[Perspective](https://perspective-dev.github.io) (an open-source data grid, pivot table and charting component,
Apache 2.0). It shows what the engine already records; it adds no forecast and no edge.

**Page:** https://claude.ai/artifact/EyLgFmNo5ZyYFsKt6Hoq3H (private to its owner unless shared from its Share menu).

## What it shows

One row per stock of the latest recorded scan (`research/scans/<date>.json`):

| Column | From |
|---|---|
| Ticker, company, industry, sector | the bank's SEC identity facts (`brain/why.py`) |
| 2-in-3 and 9-in-10 ranges, typical move | the scan's W1 ranges, over the next five sessions, either way |
| 20% day chance | the scan's S1 chance of a 20%+ session against SPY within five |
| Listed after 2010 | the scan's newer listings (ranges tested by W1b) |
| Why the range is wide | the scan's own reasons, as `scan run` prints them for the top of its list |
| Move now, where now | the latest check against that scan (the desk's case file): the move since the close and whether it sits outside a range |
| Why it moved | the check's explanation and headline for a flagged stock |

Views: moving now, outside its range, widest ranges, likeliest 20% days, by industry, a market map (when the charts
plugin loads), all columns. The desk's latest note gives one filter per theme. Tapping a row draws that stock's
ranges to scale with its move since the close.

Only numbers the repository already records go into the page's data (ranges, chances and moves in percent). No
price, volume or other Yahoo data is published.

## Commands

```
python -m brain explorer build     # var/explorer/data.json from the latest scan, check and desk note
python -m brain explorer vendor    # Perspective's browser files into var/explorer/lib (npm pack, pinned version)
python -m brain explorer files     # the published path of every file the page needs
```

The page itself is `web/explorer/index.html`. Perspective is pinned at 5.6.1 (2026-10-23) and its files are
published with the page, because the page's security rules let it fetch only its own files at runtime (the engine's
WebAssembly included). If Perspective cannot start, the page falls back to a plain sortable table of its own.

## Keeping it current

Each scheduled check (before the open, midday, after the close) ends by running `explorer build` and republishing
`data.json` to the same page. The first build after a new scan reads every stock's features for the reasons (about
three minutes); later builds that day reuse them. On a fresh machine `explorer vendor` is needed only to republish
Perspective's files, which the page already holds.

Publish with the Artifact tool, passing the page's link as `url`, `file_path` `web/explorer/index.html` and `files`
`{"data.json": "var/explorer/data.json"}`. Without `url`, a session that did not itself publish the page (after a
reset or a summary) creates a separate page with no Perspective files. The tool may first ask for the published
`data.json` to be read (`action: "read"`, `path: "data.json"`); that is the previous build, which the new one replaces.
