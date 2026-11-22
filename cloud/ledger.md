# Compute ledger

Budget: $50. Prices are Spot rates per chip-hour from the Cloud Billing catalog on 2026-11-10; totals are
estimates from wall-clock time and exclude boot disks, which add a few cents per session.

| Session | Date (UTC) | Machine | Zone | Rate | Duration | Estimate | Purpose |
|---|---|---|---|---|---|---|---|
| Probe | 2026-11-10 06:50 | ct6e-standard-1t | europe-west4-a | $1.78/h | 2 min | $0.06 | Quota and capacity check |
| 1 | 2026-11-10 07:53 to 09:26 | ct6e-standard-1t | europe-west4-a | $1.78/h | 1.55 h | $2.76 | Single-chip study: operator coverage, execution modes, expert implementations, proxy models |
| 2, attempt 1 | 2026-11-10 09:48 to 09:51 | ct6e-standard-4t | us-east1-d | $2.82/h | 3 min | $0.14 | Preempted during bootstrap |
| 2, attempt 2 | 2026-11-10 10:00 to 10:11 | ct6e-standard-4t | us-east1-d | $2.82/h | 11 min | $0.52 | Released DiffusionGemma on four chips at batch 1 and 4, then preempted |
| 2, attempt 3 | 2026-11-10 11:20 to 11:30 | ct6e-standard-4t | us-east1-d | $2.82/h | 10 min | $0.47 | Autoregressive baseline; preempted while compiling |
| 2 | 2026-11-10 11:36 to 12:47 | ct6e-standard-4t | us-central1-b | $2.82/h | 1.18 h | $3.34 | Released weights for both models: batch ladder, default-experts comparison, GSM8K, stop-check variants; preempted in the last stage |

Running total: about $7.29.

## Spot price per v6e chip-hour by region

| Region | Spot | On-demand |
|---|---|---|
| us-east4 | $0.24 | $2.70 |
| asia-southeast1 | $0.27 | $2.97 |
| europe-west2 | $0.29 | $3.24 |
| us-south1 | $0.54 | not listed |
| us-central1, us-east1, us-west1 | $0.70 | $2.70 |
| southamerica-west1 | $1.03 | $2.70 |
| asia-south1 | $1.56 | $3.24 |
| asia-northeast1 | $1.59 | $3.24 |
| us-east5 | $1.62 | not listed |
| europe-west4 | $1.78 | $2.97 |

The project has a quota of 48 v6e chips in most regions and 0 in us-east4 and us-east5.
Requests for 8 chips in us-east4 and us-central1 were filed on 2026-11-10 and denied.
On-demand quota is 0 in us-central1, us-east1 and us-west1, where only Spot is usable.
