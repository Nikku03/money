# Customers: who a company sells to, as its annual report says

Asked for on 2026-11-25: "who it trades with". A company must say in its annual report when one customer brings a
tenth or more of its revenue, and many name that customer. This layer of the company file (`docs/FILE.md`) reads
those sentences.

```
python -m brain customers read --broad           # the latest 10-K of each company, from SEC EDGAR (about 10 minutes)
python -m brain customers read --band small      # or --band micro, or tickers by name
python -m brain customers build                  # the library, with who names whom (var/customers/, seconds)
python -m brain customers show SWKS WMT          # who it sells to; who names it as a customer
python -m brain customers read --broad --cached  # after a new reader: again from the kept passages, without SEC
```

## What is read (`brain/customers.py`)

Fixed rules, no model. From the report's text the reader keeps the passages with a percentage, a revenue word and
a stated share, and from those:

| Piece | Example from a report |
|---|---|
| a named customer and its share | "Apple ... in the aggregate accounted for 61%, 67%, and 69% of the Company's net revenue" |
| several, each with its share | "Cisco Systems, Inc., NVIDIA Corporation, Nokia Corporation and Amazon.com, Inc. contributed 19.9%, 16.3%, 10.7% and 10.5%, respectively" |
| a customer table | "Toyota Motor Company Volkswagen Group General Motors 2025 18% 11% 10%" |
| the largest unnamed share and how many | "sales to one direct customer represented 22% of total revenue and sales to another direct customer represented 14%" |
| the top five or ten | "our five largest customers accounted for approximately 37% of our net revenue" |
| the government | "72% of our $75.0 billion in sales were from the U.S. Government" |
| none | "No sales to an individual customer ... accounted for more than 10% of revenue" |

**A name counts only where a customer stands.** A name is recognised when it is a company of SEC's ticker list
(its 2,500 largest, by full name; a one-word name only when no other listed company starts with that word) or one
of a short list of well-known names. It is taken as a customer when it comes after "sales to" or "revenue from",
before its own share, in a row of a customer table, or in a list right after the word "customers". It is not
taken with "products", "brands", "segment", "business" or "acquisition" after it, nor in a sentence about
vendors, brands sold, credit ratings, total assets, a product's sales or units sold.

**Which percentage.** Reports list the latest year first: a single customer takes the first of its percentages
(the last when the years run upwards). Names that run together take the percentages after them in order when
there are as many and the sentence says "respectively". A share of a group ("four customers, including PACCAR,
were 31%") is no one's own. "10% or more" is the reporting threshold, not a share. A starred table cell means
below a tenth in the latest year: that customer is not named.

Each reading is one `customers.major` fact with the filing's URL and up to eight of the sentences themselves.
The card quotes the report. The candidate passages of each report are kept in `var/customers/chunks/`, so a later
reader can read again without SEC.

## What it found (the research universe, latest annual reports, read 2026-11-26)

| Of 741 companies | |
|---|---|
| state a largest customer share | 149 (median 19%; 54 of them a quarter of revenue or more) |
| say how many customers bring a tenth or more | 191 |
| name a known customer | 62 (60 a company listed in the United States): 104 links, 77 with a share |
| give their five largest customers' share | 25 |
| give their ten largest | 50 |
| give the U.S. government's share | 29 |
| say no customer brings a tenth | 93 |
| nothing read | 434 |

The most named: Walmart (by 15 companies), AT&T (6), Apple (5), McKesson, Cencora, Verizon and Amazon (4 each).

## How well the reader reads

- The rules were built by reading. The first reader took any known name in a customer sentence: of the 78
  companies it found, about one in five was wrong (a stock exchange, a credit-rating agency, the brands a dealer
  sells, a vendor, the company's own segment). Reader 4 was checked by reading every one of its 104 links against
  its sentence: none of those errors is left among them. Reader 5 added what a sample of the older reports
  showed (a subsidiary's share, a growth figure, a spun-off parent).
- It misses more than it invents. A customer named but not listed in the United States (a distributor, Samsung)
  stays in the quoted sentence only. A report that says "one customer" names no one. A wording the rules do not
  know is not read: 434 of 741 reports gave nothing, and most of those have no customer of a tenth, but not all.
- Known limits: a share stated for a subsidiary or for a product can still be read as the company's when the
  sentence does not say so; a company named beside a share may be a partner whose payments are revenue (Regeneron
  and Sanofi); the latest report only, so a link is as of that report's date.

## Every year's report, and links by date

```
python -m brain customers read --broad --all-years    # every annual report since 2010 (about two hours; resumes)
python -m brain customers build --all-years           # the history: one row a report (var/customers/history.json.gz)
```

- A link is known from the moment the report naming the customer was filed, until the supplier's next report (or
  456 days). `customers.links` gives them: 2,271 links of 179 suppliers to 147 listed customers from the 12,169
  annual reports of the research universe since 2010 (reader 5).
- The read first adds the older annual reports to each company's filing list (`customers.older_reports`), and
  only those: a large bank's older pages hold over a hundred thousand note prospectuses.
- A sample of 45 pairs read against their sentences: two were not customers (a spun-off parent, a customer of
  another sentence), three were a subsidiary's or a segment's customer. Reader 5 leaves those five out. The
  names are those of companies listed today: a customer since bought or delisted is not in the links.

## On the old years (customers' reports of 2011-2023), before any test

`research/customers/explore_train.py` and its log. An event is a results report of a customer while a link holds;
events where the supplier itself reports nearby are left out. Each number is the rank correlation of the
customer's report-session move with what the supplier did, inside a season, averaged over the seasons.

| What the supplier did | Linked suppliers | Industry peers that do not name the customer |
|---|---|---|
| the same session | +0.220 (t +6.6, 26 of 30 seasons) | +0.075 (t +2.2) |
| the next 5 sessions | +0.040 (t +1.4) | -0.012 |
| the next 21 sessions | +0.035 (t +1.3) | +0.015 |
| its own report, 1 to 45 days later | -0.028 (t -0.9) | -0.023 |

- **On the day the link is real.** After the customers' best fifth of reports (+7.32%), their suppliers moved
  +0.61% the same session and 64% rose; after the worst fifth (-7.71%), -0.70% and 37% rose. About a tenth of the
  customer's move.
- **After the day, little.** The next five sessions gave +0.099 on 2011-2018 and +0.010 on 2019-2023: what was
  there has faded.
- The supplier's own later report: nothing (not declared).

## The declared tests (`brain/research/customertest.py`)

```
python -m brain customertest declare      # after committing the code
python -m brain customertest test         # once; then commit the ledger and research/RESULTS.md, whatever they say
```

TEST is customers' reports of 2024-01..2026-09 (1,166 before the supplier's own reports are left out; no outcome
of those years was read before the declaration). The link set is pinned by its digest.

- **C1a (the day).** A supplier that names the customer moves with its report more than its industry peers that
  do not: the linked correlation less the peers', per season. About the same session: it says the links are real
  and explains a move. It is not a forecast.
- **C1b (after).** The customer's report-session move tells the supplier's next five sessions. Eleven seasons of
  about a hundred events can tell an effect of the size seen on 2011-2018 (+0.10) from none; they cannot tell
  +0.04 from none.

Run once on 2026-11-26 (`research/RESULTS.md`, section "Customers: who it sells to"): 694 reports of a customer
while a link held (118 links, 52 customers), eleven seasons.

| Test | 2024-01..2026-09 | p | BH q | Verdict |
|---|---|---|---|---|
| C1a the day: the link adds to the same-session correlation | +0.062 (linked +0.109, industry peers +0.047), positive in 7 of 11 seasons | 0.10 | 0.30 | **FAIL** |
| C1b after: the supplier's next five sessions | -0.093, positive in 2 of 11 seasons | 0.98 | 1.00 | **FAIL** |

**What the two FAILs mean.**
- **The day.** On 2011-2023 a named supplier moved with its customer's report clearly more than its industry
  (the link added +0.156). On 2024-2026 it added +0.062 and did not clear the bar. After the customers' best
  fifth of reports (+7.87%) their suppliers moved +0.00% the same session and 48% rose; after the worst fifth
  (-8.34%), -0.36% and 41% rose. The read-through on the day has become small.
- **After.** Nothing to follow: the suppliers' next five sessions went, if anything, the other way (-0.093; the
  next 21 sessions -0.027; industry peers -0.060). That reversal was not declared and is not a finding.
- So a link says who a company depends on. It did not say what its stock does when the customer reports.
  Do not present a read-through from a customer to its supplier as an edge, on the day or after.

## What it is not

- The card shows each company's latest report. The history behind the tests is a separate file.
- Suppliers are seen only from the other side: a company's suppliers are the companies that name it as a customer.
- The small, micro and newer-listing universes are not read yet.
