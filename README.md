# E-Commerce Customer Segmentation & Automated Retention Engine

This project is an end-to-end retention system built on real transactions from a UK online giftware retailer (the UCI Online Retail II dataset, about 1.07M invoice lines from Dec 2009 to Dec 2011). It cleans and loads the raw data into DuckDB and models it with dbt into a star schema. K-Means then segments customers on RFM (recency, frequency, monetary). A local CrewAI agent running on Ollama drafts retention emails for the segments that are slipping away. A Streamlit app ties it together, and a weekly GitHub Actions job rebuilds everything from the source data and emails an alert if anything breaks.

**Business question this answers:** which customers drive the revenue, which valuable customers are drifting away, and what should we send them to win them back, without discounting to people who'd buy anyway?

## Key findings

- **Revenue is highly concentrated.** The 1,189 "Loyal Champions" (20% of active customers, median 13 orders, last ordered 17 days before the snapshot) account for **73.6% of net revenue**.
- **£3.29M of lifetime value sits in the two churn-risk segments.** "At-Risk Whales" are 1,440 customers with a median of 184.5 days since their last order (£2.68M lifetime). "Dormant Buyers" are 1,950 mostly one-time buyers, with a median of 402 days since their last order (£0.61M).
- **Champions lapse quickly.** In the simulator, if no one orders for 90 days, the trained model re-scores 537 of the 1,189 Champions as At-Risk Whales. A retention program needs to watch Champions' recency, not only customers who have already lapsed.
- Net revenue over the last 12 full months (Dec 2010–Nov 2011) was £7.93M, down 2.0% from £8.09M in the prior 12. Both years peak in Oct–Nov ahead of Christmas.

## Architecture

1. **Fetch:** `pipelines/fetch_data.py` downloads the dataset from the UCI repository and writes one CSV per workbook sheet to `data/raw/`.
2. **Load & clean:** `pipelines/load_to_duckdb.py` writes `retail_analytics.duckdb`. Every raw row ends up in exactly one place: `raw_transactions` (clean sale lines), `raw_returns` (cancelled lines by known customers), or `audit_rejected_rows` with a reason. The rules run in order, and the first one that matches wins:
   - unparseable values
   - the **2010-12-01..09 overlap**: both UCI sheets contain those 9 days (1,088 invoices, 22,523 rows), so an invoice that already appeared in the earlier file is dropped from the later one
   - exact duplicate lines
   - `A` bad-debt adjustments
   - non-product codes (postage, Amazon fees, manual adjustments, samples, test items, vouchers)
   - missing customer ID
   - non-positive price
   - negative quantity without a `C` invoice

   The load runs in one transaction and checks that the row counts reconcile before it commits.
3. **Transform (dbt-duckdb):** staging views, then a star in the `marts` schema:
   - `fct_invoice_lines`: the fact, one row per sale line.
   - `dim_customers`, `dim_products`, `dim_date`.
   - `fct_customer_transactions`: one row per customer with **recency_days, frequency (distinct orders), monetary (net of returns)**, plus AOV, return rate and tenure.
   - `agg_monthly_sales`: the sales-velocity series.

   43 dbt tests, including four reconciliation checks in `data_tests/`.
4. **Segment (scikit-learn):** `ml/segmentation.py` log1p-transforms RFM, applies `StandardScaler`, and picks k from {4, 5, 6} by silhouette (it chooses k = 4). It names clusters by matching each centroid to a persona prototype in (recency, value) space using the Hungarian algorithm. It writes `marts.dim_customer_segments` plus segment profiles, k diagnostics, and the scaler and centroids, so the app can re-score what-if data without a pickled model.
5. **Local AI agent:** `agent/retention_agent.py` runs a CrewAI "Lifecycle Marketing Analyst" on Ollama in two tasks:
   1. Call the `@tool`s, which run fixed, read-only SQL against DuckDB (cohort size, RFM medians, revenue share, top products, countries, recency bands), and write analyst notes on why the cohort is drifting and what will resonate.
   2. Write the campaign: strategy, A/B subject lines, preview text, email body, and follow-up.

   The facts and the copy are kept apart on purpose. The draft's **Cohort snapshot** section is rendered in Python straight from the warehouse, so a small model can't misquote the numbers. A per-segment incentive policy caps discounts: 15% for Whales, 20% for Dormant, 10% for Newcomers, 5% for Champions. A task **guardrail** checks the draft for the required sections, two subject lines, the `*|FNAME|*` merge field, at least 80 words of email body, and the discount cap. It sends failures back to the model with the specific problems, up to 2 retries. After that, the draft is kept but flagged as failed rather than discarded. Echoed tool-call JSON and stray code fences are cleaned out, and drafts are saved to `campaigns/`.
6. **App:** `app/main.py` is a Streamlit app with three tabs:
   - **Executive Cohort & LTV:** KPIs, revenue concentration, segment profiles, monthly velocity, and new vs returning customers.
   - **AI Retention Campaigns:** the segment brief, a generate button, and the guardrail status.
   - **Churn Risk Simulator:** a lapse-threshold slider, plus a "fast-forward N days" slider that re-scores every customer with the trained model and shows where they move.
7. **Monitor:** `tests/test_portfolio_health.py` holds 30 checks against the real warehouse. `pipelines/notify_failure.py` sends a Gmail SMTP-over-SSL alert with the failing step's log tail and a link to the run. `.github/workflows/health_check.yml` rebuilds steps 1–4 and runs pytest every Monday at 08:00 UTC (and on demand). `ci.yml` runs the unit tests on every push.

Requires Python 3.13. `dbt-duckdb`'s dependency chain has no CPython 3.14 wheels yet (same constraint as `../B2B_CRM_Pipeline`).

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Data -> warehouse -> segments (about 2 minutes, mostly parsing the UCI workbook once)
.venv/bin/python pipelines/fetch_data.py
.venv/bin/python pipelines/load_to_duckdb.py
DBT_PROFILES_DIR=. .venv/bin/dbt build
.venv/bin/python ml/segmentation.py

# Tests (unit + health checks against the built warehouse)
.venv/bin/pytest -q

# Local AI agent: needs Ollama running with a pulled model (default llama3.1)
ollama pull llama3.1
.venv/bin/python agent/retention_agent.py --all-risk      # or --segment "At-Risk Whales"
# OLLAMA_MODEL=llama3.2 (or mistral) uses a different local model

# App
.venv/bin/streamlit run app/main.py
```

**Email alerts:** in the GitHub repo, add three Actions secrets:

- `ALERT_SENDER_EMAIL`: a Gmail address.
- `GMAIL_APP_PASSWORD`: a [Gmail App Password](https://support.google.com/accounts/answer/185833) for that address. This is not the account password, and it requires 2-Step Verification.
- `ALERT_RECEIVER_EMAIL`: where alerts go. Comma-separate several addresses.

To test the message locally without sending anything: `python pipelines/notify_failure.py --message "test" --dry-run`.

## Portfolio outputs

- **`Ecommerce_Customer_Segmentation_Analysis.html`**: the write-up, a single self-contained page (charts embedded) with findings, recommendations and data scope. `docs/Ecommerce_Customer_Segmentation_Analysis_Report.pdf` is the same page as a PDF.
- `outputs/`: 5 charts (PNG) and 5 summary tables (CSV): segment profiles, monthly sales, k diagnostics, top 25 products by net revenue and revenue by country, all written by `reports/make_outputs.py` from the marts.
- `dashboard/`: screenshots of the three Streamlit tabs (PNG, plus one combined PDF).

Rebuild them after `dbt build`:

```bash
.venv/bin/python reports/make_outputs.py
CHROMIUM_PATH=/path/to/chrome .venv/bin/python reports/build_page.py --name Ecommerce_Customer_Segmentation_Analysis   # or `playwright install chromium` and drop CHROMIUM_PATH
```

## Tools

Python (pandas, NumPy, scikit-learn, SciPy), DuckDB, dbt-duckdb, SQL, Streamlit, Plotly, CrewAI, Ollama (llama3.1 / llama3.2), pytest, GitHub Actions, smtplib.

## Data honesty

- **Only 78% of raw rows can be attributed to a customer.** 232,991 lines (21.8%) have no Customer ID; most are guest checkouts. They're kept in `audit_rejected_rows` but can't enter RFM, so the segments describe *registered* customers. Monthly revenue in the app is registered-customer revenue as well. It's not total company revenue.
- **Most customers are businesses.** Per the UCI description, many customers are wholesalers. This explains the extreme spend tail (max net spend £578k against a £840 median), and it's why RFM is log-transformed: on raw values, K-Means scores a higher silhouette (0.59 at k = 4) only by putting 4 and 37 wholesale accounts in clusters of their own. The log-scale solution has a lower silhouette (0.363) but gives four balanced clusters (1,189–1,950 customers each) that a marketer can act on. Both numbers are reported, so the trade-off is visible.
- **"Whale" is relative.** At-Risk Whales have a median net spend of £1,432. That's above this base's median, but these are mid-to-high-value lapsing accounts, not the very largest ones (most of those are Champions). The persona names come from the spec; the medians in the app show what each segment actually is.
- **Monetary is net of returns.** Cancelled lines are subtracted. The 20 customers whose returns cancel out everything they bought have net spend ≤ 0, so they're excluded from clustering ("active" = net spend > 0). 87 return lines belong to customers with no sales in the window (they bought before Dec 2009); those are left out of customer metrics rather than creating negative customers.
- **Recency is measured from 2011-12-10**, the day after the last invoice, so the data reads as a snapshot. The data starts in Dec 2009, so every customer in that month counts as "new" (left-censoring). New-customer counts are only meaningful from early 2010.
- Exact duplicate lines (11,812) are treated as repeat scans and dropped. Some could be genuine identical re-orders within one invoice. There's no way to tell them apart in this data, so they stay in the audit table.
- **The agent's copy is a draft, not a send.** Early test runs with a 3B model showed why facts and copy are now separated:
  - it labelled the 24.7% customer share as "revenue share"
  - it read the 2.3% return rate as "2.3% of customers haven't ordered in 90 days"
  - it called buyer counts "orders"
  - it put the whole email into the preview line

  The snapshot numbers now come from SQL, not the model, and the guardrail catches structural failures. The analyst notes and copy are still model judgment, which is why every draft is labelled "Review before sending".

## Testing status (2026-10-19)

- **pytest:** 88 tests, all passing.
  - Unit tests:
    - 10 for the loader, on a hand-built CSV where each line trips exactly one rule. Covers row conservation, the cross-file overlap, Online Retail I header names, and rollback leaving the previous load intact.
    - 11 for segmentation, on synthetic data with known personas. Covers >98% persona recovery, unique names, deterministic fits, the stored model reproducing the fit labels, and net returners being excluded.
    - 27 for the agent, with no LLM needed. Covers the brief tools, the discount regex, the guardrail retry-then-accept flow, heading matching, the email-body check, tool-echo cleanup, the data-rendered snapshot, the Ollama checks (missing model, server down), and building the full crew so CrewAI's validation runs.
    - 8 for the alert, with a fake SMTP server.
  - Health checks: 30 against the real warehouse.
- **dbt:** `dbt build` passes 51/51 (8 models, 43 tests). **Negative control:** re-running the recency reconciliation test with the snapshot var set to 2011-12-20 (tables built at 2011-12-10) fails on exactly 5,852 rows, every customer, as expected.
- **Health-check negative control:** on a copy of the real database, I set one segment's customer ID to NULL and deleted another customer's segment. 4 of the 30 health checks failed, including "no null customer IDs" and "every active customer is segmented". The alert dry-run put the failing assertion at the top of the log tail.
- **CI simulation:** with the database path pointed at a missing file, the push-CI command passes (51 passed, 5 DB-dependent agent tests skipped). The health suite *fails* (1 failed, 29 errors) rather than skipping, which is what should trigger the alert.
- **3× repeat run:** load → `dbt build` → segmentation, run three times into fresh databases. All 14 tables were content-identical, with run timestamps excluded. The first attempt showed two ML diagnostic tables differing by one floating-point ULP (~2e-16, from multithreaded summation in scikit-learn), even though all 5,832 cluster assignments were identical. Stored parameters are now rounded to 10 decimals, and three further runs were byte-identical.
- **Bugs found and fixed while building:**
  - The two UCI sheets overlap for 9 days, which would have double-counted 22,523 rows.
  - Labelling clusters by spend rank alone (as in the reference code this project started from) would have called the most valuable *recent* cluster "At-Risk".
  - The Streamlit app crashed on a duplicate `hovermode` layout argument, and outside bar labels were clipped. Both were found by looking at the rendered app.
  - CrewAI treated `{first_name}` merge fields as its own template variables.
  - The guardrail was rejected at task construction because `from __future__ import annotations` turns its return annotation into a string. A test now builds the crew so this kind of wiring error fails in CI.
  - The 3B model couldn't reliably copy the cohort snapshot between tasks, and misread figures when restating them. The snapshot is now rendered from SQL.
  - A draft passed every structural check with an empty email body (all the copy was in the preview line). This is now caught by the 80-word body check.
- **Agent runs (llama3.2, 3B):** five rounds of real runs drove the design changes above. In the final run, both churn-risk segments passed the guardrail on the first attempt, in about 2 minutes per segment on a laptop, and both offers stayed within policy (VIP perk with no discount for Whales; 15% for Dormant, under its 20% cap). Remaining model slips are listed in `campaigns/samples/README.md`. The unedited outputs are committed alongside it.
- **Not tested:**
  - The real Gmail send (no credentials were used here; the SMTP path is tested against a fake server).
  - The GitHub Actions workflows themselves, which run once the repo is pushed.
  - The spec's default `llama3.1` model, which isn't pulled on this machine. The agent was run with `llama3.2`.
  - Whether the campaigns actually lift retention. That needs a live send and a holdout group.

## Repo hygiene note

`data/raw/*.csv` (93 MB), `*.duckdb`, `campaigns/*.md` and `health_logs/` are gitignored. The raw data is re-downloaded from UCI by `fetch_data.py` (and cached in CI), and everything downstream is rebuilt deterministically from it, so nothing is lost by excluding them. A few example agent outputs are committed under `campaigns/samples/`. No secrets live in the repo: the Gmail credentials exist only as GitHub Actions secrets, and the local LLM needs none.

## License

Code: MIT (see `LICENSE`). Data: the [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) dataset (CC BY 4.0), downloaded at build time and not committed.
