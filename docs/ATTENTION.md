# Attention (`brain/attention.py`)

Asked for on 2026-11-23 ("give the brain attention"), after asking whether the brain can dig deeper and reason. The
desk's case file lists its movers alike, furthest outside their ranges first. Attention decides what deserves a closer
look, then digs there first, as an analyst would.

```
python -m brain desk build            # the case file for the latest check
python -m brain attention             # what deserves attention, and the digging (--top 6, --full, --no-read)
```

## What gets attention

Every mover in the case file, every stock that moved against its wave (the pond, `docs/RIPPLES.md`), and every stock
the brain was already watching (the last note's themes and open questions, open desk calls, open calls to the person)
is scored on seven parts, each from 0 to 1:

| Part | Weight | What it measures |
|---|---|---|
| surprise | 25% | how far past its 9-in-10 range the move went (or its own move in usual days) |
| unexplained | 20% | the share of the move the market and the drivers do not carry, more when no news was found |
| news | 15% | filings since the close (serious 8-K items most: accounts not to be relied on, bankruptcy, an impairment, a listing notice, a change of auditor, a restructuring, a material deal, an officer change) and stories that name it |
| reach | 10% | how many stocks moved closely with it over the year (a link of 0.45 or more) |
| against | 10% | moving against the wave its neighbours sent it |
| memory | 10% | something the brain was already watching |
| fragile | 10% | mosaic chances three times the usual stock's or more, or cash covering under four quarters of losses |

The scores become attention weights through a softmax, so they sum to 100% and the top few take most of them.

One check guards against a false alarm: if the move since the close rests on one or two prices, the score is halved,
with the reason written. Such a move is a stray print, like N-able's single late trade on Nov 20. The weights are fixed
by reasoning, not fitted: nothing here has tested which items turn out to matter.

## The digging, for the top six

- **Filings** since the close, read deeper than the case file's opening: an 8-K's press release, else the filing's own
  text, with the sentences that carry numbers, guidance or events pulled out.
- **Stories** that name it, read through the engine's web reader. A refused page is listed as refused, never got
  around.
- **Memory:** what the brain said or found about it before, from the memory bank, around the stock's name.
- **The pond:** the wave it felt and the stones that sent it, how it moved (with the water, against its wave, its own
  news), and who its move reaches, with their moves.

## What it is and is not

Attention says where to look. It is not a forecast and not evidence. What it reads is for the desk note to reason
about and cite: pages by their URL under `read`, case-file items by their ids.

Each check's focus is kept in `research/attention/`, so whether the items it put first turned out to matter (a story
by the next check, a move past the range) can be measured once enough checks have passed.

On Nov 20's after-hours check it ranked:

1. Kymera: an open call, its trial results due Monday.
2. Zscaler: reported takeover talks; it read the Motley Fool story; Yahoo's page refused automated reading.
3. CrowdStrike: against its wave.
4. N-able: a -12.2% move resting on a single trade, so halved.
