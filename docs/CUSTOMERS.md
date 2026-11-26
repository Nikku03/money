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
  its sentence: none of those errors is left among them.
- It misses more than it invents. A customer named but not listed in the United States (a distributor, Samsung)
  stays in the quoted sentence only. A report that says "one customer" names no one. A wording the rules do not
  know is not read: 434 of 741 reports gave nothing, and most of those have no customer of a tenth, but not all.
- Known limits: a share stated for a subsidiary or for a product can still be read as the company's when the
  sentence does not say so; a company named beside a share may be a partner whose payments are revenue (Regeneron
  and Sanofi); the latest report only, so a link is as of that report's date.

## What it is not

- Not evidence of anything about a stock. Whether a customer's news reaches its supplier's stock, on the day or
  after, is **not tested here**. The related tests on stocks that move together failed for what follows the day
  (R1b-R1d, `docs/RIPPLES.md`).
- Not a point-in-time history. Only each company's latest report is read. A fair test needs every year's report
  (about 10,000 for this universe, two hours of SEC time), so that a link is known only from the day it was filed.
  That is the next step, under the usual discipline: old years first, declare, run once.
- Suppliers are seen only from the other side: a company's suppliers are the companies that name it as a customer.
