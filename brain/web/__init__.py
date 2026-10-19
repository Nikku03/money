"""The reasoning engine's gateway to the web, with no language model (docs/WEB.md).

``gateway``: a browser (browser-use's page API, no agent) or plain HTTP, under the repo's web rules.
``extract``: fixed rules that take facts out of a page. ``events``: fixed rules that tag corporate events.
``plan``: what the engine asks for, from the bank's gaps, and what it keeps.

    brain web get URL [--links PATTERN | --tables] [--http]     read one page and show what the rules extract
    brain web plan [TICKER...]                                  what the engine would fetch, and why
    brain web run [TICKER...] [--per-ticker N] [--http]         fetch it and keep the facts
"""
from __future__ import annotations


def _tickers(words) -> list[str]:
    if words:
        return [w.upper() for w in words]
    from brain import watch
    return [t.upper() for t in watch.load().get("tickers", [])]


def _cmd(a) -> int:
    from brain.bank import Bank
    from brain.web import extract, plan
    from brain.web.gateway import Gateway, browser_available
    if a.action == "plan":
        tasks = plan.plan(Bank(), _tickers(a.words), a.per_ticker)
        for t in tasks:
            print(f"{t['ticker']:<6} {t['recipe']:<8} {t['url'][:90]}\n       why: {t['why']}")
        print(f"{len(tasks)} tasks" if tasks else "nothing to fetch: every headline in the bank has been read")
        return 0
    ok, why = browser_available()
    mode = "http" if a.http else "auto"
    if not a.http and not ok:
        print(f"no browser ({why}); reading over plain HTTP")
    with Gateway(mode=mode, max_pages=a.max_pages) as gw:
        if a.action == "get":
            if not a.words:
                print("usage: brain web get URL")
                return 2
            page = gw.open(a.words[0])
            if not page.ok:
                print(f"not read: {page.error}")
                return 1
            print(f"read {page.final_url} (HTTP {page.status or '?'}, {page.mode}, {len(page.html):,} characters)")
            for r in page.refused[:5]:
                print(f"  refused while rendering: {r['url'][:80]} ({r['why']})")
            if a.links is not None:
                for ln in extract.links(page.html, page.final_url, a.links or None)[:40]:
                    print(f"  {ln['url'][:100]}  {ln['text'][:60]}")
            elif a.tables:
                for i, tb in enumerate(extract.tables(page.html)):
                    print(f"  table {i}: {len(tb)} rows; first: {tb[0][:6]}")
            else:
                from brain.web import events
                art = extract.article(page.html, page.final_url)
                for k in ("headline", "published", "modified", "author", "publisher", "section", "description"):
                    print(f"  {k:<11} {art[k]}")
                print(f"  {'events':<11} {', '.join(events.tag(art['headline'], art['description'])) or '-'}")
                print(f"  {'text':<11} {art['chars']:,} characters; structured data: {art['structured']}")
            return 0
        bank = Bank()
        tasks = plan.plan(bank, _tickers(a.words), a.per_ticker)
        print(f"{len(tasks)} tasks through the {gw.mode} gateway")
        out = plan.run(bank, tasks, gw)
        print(f"kept {sum(k['new'] for k in out['kept'])} new web.article facts from {out['pages']} pages; "
              f"{len(out['refused'])} refused (rules, robots.txt, blocks)")
    return 0


def add_cli(sub) -> None:
    s = sub.add_parser("web", help="the reasoning engine's web gateway, no model: get URL | plan | run")
    s.add_argument("action", choices=["get", "plan", "run"])
    s.add_argument("words", nargs="*", metavar="URL | TICKER")
    s.add_argument("--links", nargs="?", const="", default=None, metavar="PATTERN", help="get: list the page's links")
    s.add_argument("--tables", action="store_true", help="get: list the page's tables")
    s.add_argument("--http", action="store_true", help="plain HTTP instead of the browser")
    s.add_argument("--per-ticker", type=int, default=5, help="plan/run: stories per ticker (default 5)")
    s.add_argument("--max-pages", type=int, default=40, help="pages the gateway may read in one run (default 40)")
    s.set_defaults(fn=_cmd)
