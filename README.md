# Kleinanzeigen Agent

Searches [Kleinanzeigen](https://www.kleinanzeigen.de) for listings and lets Google Gemini rate each one
(text **and** all images) against criteria you write in plain language. The result is a sorted list:
`Yes` → `Caution` → `No`, cheapest first.

Scraping is done by the [`ebay-kleinanzeigen-api-mcp`](https://github.com/flo-bauer/ebay-kleinanzeigen-api-mcp)
client (Playwright/Chromium), included as a git submodule.

## Setup

Requires Python 3.11+ and a [Gemini API key](https://aistudio.google.com/app/apikey).

```bash
git clone --recursive https://github.com/flo-bauer/kleinanzeigen-agent.git
cd kleinanzeigen-agent
python -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip
.venv/bin/python -m playwright install chromium
cp .env.example .env                           # then put your key into .env
```

Already cloned without `--recursive`? Run `git submodule update --init`.

## Usage

```bash
.venv/bin/python kleinanzeigen_agent_general.py --config configs/tv_defekt.json
```

The agent prints the rated listings and writes them as JSON to `output_file`.

## Config

One JSON file per search, see [`configs/tv_defekt.json`](configs/tv_defekt.json):

| Key | Meaning |
|---|---|
| `search.query` | Search term |
| `search.location` | City or postal code |
| `search.radius_km` | Search radius |
| `search.min_price`, `search.max_price` | Price range in EUR (optional) |
| `search.max_pages` | Result pages to fetch (~25 listings each) |
| `manual_listings` | Extra listing IDs to rate in addition to the search |
| `criteria` | What you are looking for: requirements, nice-to-haves, exclusions. Free text, sent to Gemini as is. |
| `output_file` | Where the results are saved (default `analysis_result.json`) |

Output per listing: `listing_id`, `title`, `price_eur`, `recommendation` (`Yes`/`No`/`Caution`), `reasoning`.

## Notes

- **Be gentle with Kleinanzeigen.** Too many requests get your IP range blocked for a while.
  Keep `max_pages` low; detail pages are fetched with only 2 in parallel on purpose.
  If you suddenly get 0 results, wait and try again later.
- Gemini models are tried in the order of `LLM_MODELS` in the script, falling back to the next one on errors.
- Unofficial tool, not affiliated with Kleinanzeigen. Use it for personal searches and respect their terms.

## License

MIT, see [LICENSE](LICENSE).
