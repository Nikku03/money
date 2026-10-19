# The web gateway: the reasoning engine online, with no language model

`brain/web` is how the reasoning engine reads the web by itself. No model is asked anything:

1. **The engine decides what it needs.** It looks at what the bank lacks and writes tasks. Each task names a recipe, a URL, the ticker and a reason.
2. **The gateway fetches.** It renders each page in a real browser (browser-use's page API, with no agent and no model), or fetches it over plain HTTP, under the repo's web rules.
3. **Fixed rules extract.** The facts come out of the page by rule, with nothing inferred. Each fact is stored in the bank with its source, the time the page says it was published, and the time it was read.

```
python -m brain web plan [TICKER...]          # what the engine would fetch, and why (default: the watchlist)
python -m brain web run [TICKER...]           # fetch it and keep the facts
python -m brain web get URL                   # read one page and show what the rules extract
python -m brain web get URL --links PATTERN   # its links (or --tables)
```

`--http` skips the browser. Without browser-use and a Chromium installed, the gateway falls back to plain HTTP by itself: `pip install "browser-use>=0.13,<0.14"`, and set `BRAIN_CHROMIUM` if Chromium isn't on the PATH.

## What the engine asks for

- **The story behind each headline.** A headline in the bank (from the news feed) whose page hasn't been read is read for what the page states about itself:
  - the headline;
  - the publication and update times;
  - author, publisher, description, section;
  - the text.

  It is kept as a `web.article` fact.
- **Listed sources.** `knowledge/web_sources.json` names pages or RSS/Atom feeds to watch for a ticker, such as a company newsroom. The gateway reads the source, keeps the links that match the source's pattern, and reads the newest few as articles. A feed is always read over plain HTTP, since there is nothing to render.

## The rules extract

- **`extract.article`:**
  - the page's structured data first (schema.org JSON-LD: `headline`, `datePublished`, `author`, `publisher`);
  - then its meta tags (OpenGraph, `article:published_time`);
  - then its text.

  A page that states no publication time gets none. Its fact is known from the time it was read, as `CLAUDE.md` asks.
- **`extract.links`, `extract.feed`, `extract.tables`:** links, feed items and HTML tables, as they are.
- **`events.tag`:** a short list of fixed patterns tags the corporate events a headline or summary names: guidance raised or cut, beat or miss, acquisition, buyback, dividend change, lawsuit, investigation, approval, rejection, upgrade, downgrade, executive change, layoffs, launch.
  - A tag says only that the words appear.
  - It implies no direction for the stock and weighs nothing.
  - What an event type does to prices is for a declared test to measure, like any other idea (`CLAUDE.md`).

## The rules it keeps

- **It says who it is.**
  - Our user agent is on every request.
  - In the browser, `navigator.webdriver` stays true. browser-use's default flag that hides automation is removed.
  - Its telemetry, captcha solver and add-on extensions are off. The add-ons are an ad blocker and a cookie-banner clicker, so with them off, pages render as they are.
  - SEC pages are read over HTTP with `SEC_USER_AGENT`.
- **Public hosts only.**
  - Every request the browser makes is checked, redirects and page assets included. A request to a local or private host is failed before it is sent.
  - Over HTTP, the connection is pinned to the checked address.
- **robots.txt and pacing.**
  - robots.txt is checked for our agent on every page and every redirect.
  - There is a pause between requests to one host, longer when the site sets a crawl-delay.
  - A run has a page budget (40 pages).
- **It stops at a login, a paywall or a bot check,** at an error status, and at anything that isn't a web page. It records why and never gets around any of them.
- **Public information only** (`CLAUDE.md`): company news, press releases and filings. Nothing about private individuals.

## What it is not

- **Not a model.** It can't decide for itself where to look, or read meaning into a page. It follows the engine's tasks and the fixed rules, so what it keeps is exactly what the page states.
- **Not evidence.** A `web.article` is a report of what a page says. It weighs nothing in a decision (`docs/DECIDE.md`) until a declared test has measured what such reports, or an event type, do to prices.
