# ClinicalTrials.gov Question → Chart Service

This is a small backend service that answers questions about clinical trials with charts as output. You ask something in plain English, like "How are Huntington's disease trials distributed across phases?", and it goes to ClinicalTrials.gov, pulls the relevant trials, counts them up and sends back a description of a chart that a frontend can draw directly.

What I cared about most is that the answers can be trusted. Every bar, point, histogram bucket, network node and edge comes with the list of trials it was built from. For each trial you get the NCT ID and the exact field in its ClinicalTrials.gov record that put it there, so any number on a chart can be checked by hand.

It handles six chart types: bar charts, grouped bar charts, time series, histograms, scatter plots and network graphs. Those cover trends over time ("how many trials started each year since 2015?"), breakdowns ("which intervention types are most common?"), comparisons between two to five drugs or conditions, geography ("which countries have the most recruiting trials?"), relationships ("show a network of sponsors and drugs"), and numeric distributions such as enrollment size or trial duration. There's also a small demo page that draws the charts, and clicking anything on it shows you the trials behind it.

## Running it

You'll need Python 3.10 or newer. From the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then put your OpenAI key in .env
python -m uvicorn ctviz.api:app --reload
```

Once it's running, open http://localhost:8000 to use the demo page

```bash
curl -s localhost:8000/v1/visualize -H 'content-type: application/json' \
  -d '{"query": "How has the number of trials for this drug changed per year since 2015?", "drug_name": "Tirzepatide"}'
```

The only setting you really need is `OPENAI_API_KEY` in your `.env` file. The model defaults to `gpt-5.4-mini`, and you can change it with `OPENAI_MODEL`. If there's no key, or OpenAI is down, the service falls back to a simpler keyword-based interpreter and still answers, and the response tells you that happened. Two other settings are useful for testing: `CTGOV_OFFLINE=true` makes it answer only from cached API responses without touching the network, and `CACHE_DIR` sets where those responses are cached (`.cache/ctgov` by default).

To run the tests, use `pytest`. There are 132 of them and they don't need a network connection. To regenerate the example outputs from live data, run `python scripts/run_examples.py`. Add `--offline` to replay the recorded API responses instead.

## Request and response format

Requests go to `POST /v1/visualize` as JSON. The only required field is `query`, the question itself (3–500 characters). Everything else is optional and lets you pin things down precisely when you don't want to rely on how the question is read:

| Field | Type | Notes |
|---|---|---|
| `query` | string | Required. The question. |
| `drug_name` | string | Brand and code names work too (`Keytruda`, `MK-3475`). |
| `condition` | string | ClinicalTrials.gov expands synonyms, so "breast cancer" also finds "Breast Neoplasms". |
| `sponsor` | string | The lead sponsor only, not collaborators. |
| `country` | string | Trials with at least one site in that country. |
| `trial_phase` | string or list | `"PHASE3"`, `"Phase 3"`, `"3"` and `"phase iii"` all work. |
| `status` | string or list | For example `"RECRUITING"` or `"not yet recruiting"`. |
| `study_type` | string | `INTERVENTIONAL`, `OBSERVATIONAL` or `EXPANDED_ACCESS`. |
| `start_year`, `end_year` | integer | Between 1990 and 2040, inclusive, and the start can't be after the end. |
| `max_records` | integer | Default 3000 (range 50–10000). The most trials fetched for each drug or condition in the question. |
| `max_citations_per_datum` | integer | Default 10 (range 0–1000). How many citations to include per data point. The full count is always reported. |
| `top_n` | integer | Only keep the N largest categories, or N nodes in a network. |

Unknown fields are rejected rather than ignored, so a typo like `drugname` gets you a clear error instead of a silently wrong answer. If a structured field disagrees with the question, the field wins, and the response mentions the override.

The response has two parts: `visualization`, which is the chart itself, and `meta`, which explains how it was produced. Here's a trimmed example for a drug comparison:

```jsonc
{
  "visualization": {
    "type": "grouped_bar_chart",
    "title": "Trials by phase: Semaglutide vs Tirzepatide",
    "subtitle": "Semaglutide: 668 trials, Tirzepatide: 250 trials",
    "encoding": {
      "x":      {"field": "phase",       "type": "ordinal",      "title": "Phase", "sort": ["Early Phase 1", "Phase 1", "..."]},
      "y":      {"field": "trial_count", "type": "quantitative", "title": "Number of trials", "unit": "trials"},
      "series": {"field": "cohort",      "type": "nominal",      "title": "Drug"},
      "tooltip": ["phase", "cohort", "trial_count", "share_of_cohort"]
    },
    "data": [
      {"phase": "Phase 3", "cohort": "Semaglutide", "trial_count": 157, "share_of_cohort": 0.235,
       "citation_count": 157, "citations": ["..."]}
    ]
  },
  "meta": {"...": "..."}
}
```

The `type` tells a frontend which kind of chart to draw, and `encoding` tells it which field in each data row goes on which axis. I used Vega-Lite's names for field types (`nominal`, `ordinal`, `quantitative`) so it maps straight onto Vega-Lite, but nothing ties it to that library. Rows arrive already sorted in the order they should be displayed. Every row has a `trial_count`, a `citation_count` and a `citations` list. The other fields in a row depend on the chart:

| Chart type | Fields in each row |
|---|---|
| `bar_chart` | The category (for example `phase`, `country`, `lead_sponsor` or `drug`) and `trial_count`. |
| `grouped_bar_chart` | The category, `cohort` (which drug or condition), `trial_count`, and `share_of_cohort`, a 0–1 fraction that's useful when cohorts are very different sizes. |
| `time_series` | `start_year` as an integer, `trial_count`, and `series` when there's more than one line. Years with no trials are filled in with zero. |
| `histogram` | `bin_label` (like `"250–499"`), `bin_start`, `bin_end` (`null` for the open-ended last bin) and `trial_count`. |
| `scatter_plot` | `nct_id`, `title`, and the two numeric values. Each point is one trial. |
| `network_graph` | Here `data` is an object with `nodes` and `edges`. Nodes have an `id` like `"drug:temozolomide"`, a `label`, an `entity_type` and a `trial_count`. Edges have a `source`, a `target` and a `trial_count`, which is the number of trials linking the two. |

A citation looks like this:

```json
{
  "nct_id": "NCT04657003",
  "title": "A Study of Tirzepatide (LY3298176) in Participants With Type 2 Diabetes Who Have Obesity or Are Overweight",
  "url": "https://clinicaltrials.gov/study/NCT04657003",
  "evidence": [
    {"field": "protocolSection.statusModule.startDateStruct.date", "value": "2021-03-29"},
    {"field": "protocolSection.armsInterventionsModule.interventions[0].name", "value": "Tirzepatide"}
  ]
}
```

The first piece of evidence explains why the trial is in this particular data point: it's counted under 2021 because its start date is in 2021. Anything after that shows why the trial belongs in the question at all: it's listed because it really does test tirzepatide. For "not reported" buckets, like trials with no phase, the value is `null`, meaning the field is missing from the record and that's exactly why the trial landed there.

The `meta` section is there so nobody has to guess what the service did. It records which interpreter handled the question (the LLM or the keyword fallback, and why), the final plan it ran, and any corrections that were made. For each drug or condition it shows how many trials ClinicalTrials.gov matched, how many were fetched and analyzed, and how many were excluded, with example IDs. It also includes plain-English notes on how things were counted, any warnings (for example when results were capped), the exact API URLs used, and the date of the data snapshot.

Bad input gets a `422`, a ClinicalTrials.gov outage gets a `502`, and offline mode without a cached response gets a `503`. A question that matches no trials isn't treated as an error: you get an empty chart and a warning explaining why.

## How it's designed, and the trade-offs

A request flows through a straight pipeline. The question is interpreted into a plan. The plan is turned into ClinicalTrials.gov queries. The results are cleaned up, checked, counted, and finally shaped into a chart. The whole design comes from one rule: **the language model is only allowed to interpret the question. Ordinary, testable code produces every number.**

The interpreter works by filling in a form rather than answering. OpenAI gets the question and has to respond by calling a single function whose parameters are the plan: what kind of analysis to run, which filters to apply, what to group by, and what to compare. Most of those fields are fixed lists of choices, so the model can only ask for things the code actually knows how to do. The response is validated with Pydantic. If it's invalid, the errors go back to the model for one more try. If that also fails, or OpenAI isn't reachable at all, a keyword-based planner takes over. After that, a second step checks the plan still makes sense as a whole. A "comparison" with only one thing to compare becomes a normal breakdown, for example, and every change like that is recorded in the response. The cost of this approach is flexibility: the service can't answer a question the plan format doesn't cover, like "median enrollment by sponsor", and adding one means writing code, not tweaking a prompt. I think that's worth it in exchange for never showing a made-up number.

Instead of asking ClinicalTrials.gov to count things for me, the service downloads the matching trials and does the counting itself. Counts alone can't come with citations, can't be checked for bad matches, and can't be turned into networks or histograms. The downside is speed, so there's a cap of 3,000 trials per drug or condition by default. Very broad questions like "all cancer trials" get analyzed on a sample, and the response says so clearly.

Inside the counting code, every number is literally a set of trial IDs: a bar's height is just the size of its set. That means a count and its citations can never disagree, because they're the same thing. To make it work, the cleanup step remembers, for every value it extracts, the exact place in the original JSON it came from and the original text.

I also didn't want to trust the registry's search blindly. When you search for a drug, ClinicalTrials.gov also returns trials that only mention it in passing, in a description or as "placebo for X". For pembrolizumab that was about 16% of results. So a trial only counts toward a drug if its intervention list, a registered synonym or its title actually names the drug. Anything excluded is listed in the response so it can be audited. The trade-off is that a trial with a typo in its record ("Semagludtide", which is real) gets left out. I'd rather be precise than inflate the numbers. I didn't apply this to conditions, because there the registry's medical synonym matching is genuinely useful.

Drug names needed their own cleanup too, because real records say things like `"Pembrolizumab 25 MG/ML [KEYTRUDA®]"`, `"lenvatinib plus pembrolizumab"` or just `"MK-3475"`. The code strips out doses and routes, splits combinations into separate drugs, drops placebos, and maps brand and code names to the generic name. This matters most for drug networks, where "Keytruda" and "pembrolizumab" need to end up as the same node.

To avoid a pile of special cases, everything you can group by (phase, country, sponsor, drug and so on) is declared once in a single registry. Each entry says which API fields it needs, how to pull the value out of a trial, and how to label and order it. The chart code never needs to know which dimension it's dealing with, so adding a new one is a single entry.

I was careful that the numbers add up. A trial registered as both Phase 1 and Phase 2 gets its own "Phase 1/2" bar instead of being counted twice, so the phase bars always add up to the number of trials. Country counts are counts of trials, not of individual sites. When you ask about recruiting trials by country, only sites that are actually recruiting count. Each response spells out whichever of these rules applied.

Networks needed some thought because they get big quickly. A sponsor–drug network for glioblastoma has around 420 connected entities, far too many to draw. Rather than picking the top nodes one by one, which can leave you with nodes that aren't connected to anything, the graph is built up from its strongest links until it reaches 25 nodes. That keeps important hubs like temozolomide (in 44 trials) and guarantees every node shown has at least one connection.

On the infrastructure side I kept things deliberately simple: synchronous Python, a cache stored as files on disk, and a rate limiter within the process. The real bottleneck is ClinicalTrials.gov's limit of roughly 50 requests a minute, not the server, so async code or Redis wouldn't buy much yet. The file cache also doubles as the set of recorded responses that make the tests and examples reproducible. The catch is that it assumes one server process. Running several would need a shared cache and a shared rate limit.

Finally, I kept the output independent of any one charting library instead of returning a Vega-Lite spec directly, so whoever builds the frontend isn't locked in. Borrowing Vega-Lite's vocabulary keeps the translation easy. The demo page does it in about 30 lines.

## Limitations and what I'd do with more time

The biggest weakness is how names are matched. The list of drug synonyms is small and hand-written; with more time I'd look drugs up in RxNorm so brand, code and salt names always merge properly. Sponsors and conditions aren't merged at all right now. "Merck Sharp & Dohme LLC" and "Merck Sharp & Dohme Corp." count as different sponsors, and "Stage IIIA Melanoma" isn't rolled up into "Melanoma". Fuzzy sponsor matching and the API's MeSH hierarchy for conditions would fix most of that.

The 3,000-trial cap means very broad questions are answered from a sample. The proper fix is to load ClinicalTrials.gov's bulk data download into a local database like Postgres or DuckDB every night and query that instead. There'd be no rate limits, no caps, and faster answers.

The keyword fallback is fairly basic. It handles the common question shapes, but it only picks one subject out of the question, so it won't notice a sponsor or country mentioned in passing. That's part of why the structured fields exist. Also, asking for "Phase 1/2" trials currently returns trials in Phase 1 or Phase 2, because the API doesn't have a filter for exactly Phase 1/2. That could be fixed with a filter after the search.

The service answers one question with one chart. It doesn't handle follow-ups like "now split that by sponsor type", or answers that need several charts. For production use I'd run multiple workers with a shared Redis cache and rate limiter, and fetch the drugs being compared in parallel rather than one after another. And while trend charts warn that recent years include trials that haven't started yet, they don't say how many, which would be a small addition.

## Example runs

The `examples/` folder has six example questions along with the real JSON the service returned for each. Each file contains both the request and the full response. All six were interpreted by `gpt-5.4-mini` and answered from ClinicalTrials.gov data as of 25 September 2026.

| File | Question | What came back |
|---|---|---|
| `01_trend_tirzepatide` | How has the number of trials for this drug changed per year since 2015? (with `drug_name: Tirzepatide`) | A time series from 250 verified trials; 39 of the 289 search results didn't actually test tirzepatide. |
| `02_distribution_phases_huntington` | How are Huntington's disease trials distributed across phases? | A bar chart of 302 trials across 9 phase groups, adding up to 302. |
| `03_comparison_semaglutide_vs_tirzepatide` | Compare phases for trials involving semaglutide vs tirzepatide. | Grouped bars comparing 668 semaglutide and 250 tirzepatide trials. |
| `04_geographic_dmd_recruiting` | Which countries have the most recruiting trials for Duchenne muscular dystrophy? | 59 trials; the US leads with 27, then Belgium (10), Italy (9) and the UK (9). |
| `05_network_sponsor_drug_glioblastoma` | Show a network of sponsors and drugs for recruiting glioblastoma trials. | A network of 25 nodes and 25 links from 328 trials, with temozolomide at the centre. |
| `06_histogram_enrollment_alzheimers_phase3` | What is the distribution of enrollment sizes for Phase 3 Alzheimer's trials? | 373 of 385 trials across 8 size ranges; the most common is 250–499 participants. |

The raw API responses behind these are saved in `examples/recordings/`, which lets the tests replay them without a network connection. One small note: replaying with `--offline` uses the keyword interpreter, which searches for "Alzheimer's" rather than the model's "alzheimer's disease", so example 6 replays with 340 trials instead of 385.

## Tools, validation, and what was generated

I built this with the help of Claude. It helped explore the API, code, tests and documentation, and review the work. At runtime the service uses OpenAI's `gpt-5.4-mini` to interpret questions. The rest is Python with FastAPI, Uvicorn, Pydantic and httpx, tested with pytest and respx and linted with ruff. The demo page uses Vega-Lite and d3.

