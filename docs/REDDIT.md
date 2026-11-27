# Reddit: public posts counted by company and topic

Asked for on 2026-11-27: "try tracking linkedin and twitter and reddit of company employees, to see if we get
any info out of that." What was built keeps to the person's own rules: companies, not people.

- **No person is followed.** No account is listed or watched, and no username or post text is stored. A post is
  found because it names a company, never because of who wrote it.
- **LinkedIn and X are not read.** Both hide posts behind a login and forbid automated collection in their terms,
  and this repo does not get around logins.
- **Reddit is read only through its official API**, with the person's own app key. The session's network proxy
  adds the key to requests to `www.reddit.com`; it never appears in the code, a file or a log.
- **Leaked material is not stored.** Only topic tags are kept, never the text. If a post looks like it reveals a
  company's confidential information, it is not a piece of evidence (CLAUDE.md, hard rules).

```
python -m brain reddit check                      # is a key saved, and does Reddit accept it?
python -m brain reddit collect [TICKERS] [--universe us_broad] [--t week] [--pages 2]
python -m brain reddit show TICKER [--days 30]    # one company: posts by day, topic and forum, with links
python -m brain reddit top [--days 7]             # the companies with the most topic mentions
```

## Getting the key

1. Log in to Reddit and open https://www.reddit.com/prefs/apps (or old.reddit.com/prefs/apps).
2. "create another app…": name `money-brain research`, type **script**, redirect URI `http://localhost:8080`.
3. The **client ID** is the short string under the app's name; the **secret** is beside "secret".
4. Save them through the session's secure form for the host `www.reddit.com`: the client ID as the username, the
   secret as the password. Never paste them into the chat.

Since November 2025 Reddit approves new API apps by hand. If the page refuses (on 2026-11-27 it said "Max clients
have been reached"), first look under "developed applications" for an app already there (a script app can be
used, or an unused one deleted). Otherwise apply through Reddit's help centre (search "Data API" or "Developer
Platform & Accessing Reddit Data") and describe the use plainly: personal, non-commercial research that counts
how often public companies and topics come up; no usernames stored, no resale, no AI training. Until a key is
saved, `reddit check` says so and `collect` does nothing.

## What it does

For each company it runs Reddit's own site-wide search, newest first, over the last week (two pages of up to 100
posts). The search words are the company's common name in quotes, with its ticker when the ticker is not an
ordinary word: `"Costco" OR COST`, `"Analog Devices" OR ADI`. A post is kept only when its title or text names
the company: the name as words, `$TICKER`, or the ticker in capitals.

Names are conservative. The first word of a company's name is used alone only when no other listed company's
name starts with it and it is not an ordinary word, a place or a person's name (`AMBIGUOUS`: "Analog" Devices,
"Walt" Disney, "Delta" in options talk). A few common names are set by hand (`ALIASES`: Google, Disney, Apple,
FedEx). Some names count only with a capital (`CASED`: "Meta", not "meta"). A ticker that is an ordinary word
("NOW", "BOX") counts only as `$NOW`. A bad name adds noise to a company's counts; it cannot point at a person.

Kept per post: its id, the forum, its time, its score and comment count, whether the forum is an investing forum,
and the topics fixed rules found in its title and text:

| Topic | Examples the rules catch | Left out on purpose |
|---|---|---|
| layoffs | layoffs, laid off, job cuts, RIF (in capitals), severance | "rif" in lower case |
| hiring_freeze | hiring freeze, offer rescinded | |
| reorg | reorganisation, restructuring | |
| outage | outage, servers down, the app is down | "the stock is down" |
| recall | a recall of, voluntary recall, defects | "I recall" |
| delay | the launch was delayed, delayed the release | "my flight was delayed" |
| strike | on strike, strike vote, walkout, unionise, picket | an option's strike price |
| closures | closing 150 stores, store closures | |
| leader_leaving | the CFO is stepping down | |
| morale | return to office, RTO, bonus cut, pay cut, PIP | |

The forums where a company's staff talk (its own forum, careers forums) are reached because the search is
site-wide. `show` splits each company's posts into investing forums and the rest.

## What it is not, yet

Nothing here is tested. Reddit's search reaches back only a short way, so there is no old record to explore. The
record runs forward from the first collection (`var/reddit/posts.jsonl`, outside git). The plan, under the
repo's usual discipline:

1. Collect at every after-close check (`brain reddit collect`, about 15 minutes for the research universe).
2. After about eight weeks, look at what a jump in a company's layoff, closure or reorganisation talk looks like
   against its own usual weeks.
3. Declare a test before looking further: does such a jump come before the company's restructuring filings (the
   mosaic's P3 picture), or before its stock's moves, more than for matched companies? Judge it on the weeks
   after.

Until that test has a result, a count is a description of what people wrote, not a forecast. This repo's history
says public talk is usually priced fast (the news-drift tests E1 and E2 failed); the hope is only in topics few
investors read.
