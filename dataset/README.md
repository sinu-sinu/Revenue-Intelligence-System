# Dataset - ACCEPTED FOR PHASE 1A

> **Status**: Production dataset for ML pipeline  
> **Review Date**: December 2024  
> **Decision**: Approved for Phase 1A implementation

---

## Why This Dataset Was Accepted

### Data Quality

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Clear prediction target | OK | Won/Lost binary outcome on 6,711 closed deals |
| Clean leakage boundaries | OK | `close_date`/`close_value` NULL for open deals |
| Sufficient labeled samples | OK | 63.2% Won / 36.8% Lost — balanced classes |
| Temporal split viable | OK | 15-month span (2016-10 to 2017-12) |
| Joinable to reference tables | OK | Links to accounts, products, sales_teams |

### Feature Availability

| Feature | Available | Source |
|---------|-----------|--------|
| `sales_agent` | OK | Direct (30 unique reps) |
| `product` | OK | Direct (7 SKUs, requires GTXPro fix) |
| `account` | OK | Direct (85 companies, 16% NULL → "Unknown") |
| `engage_date` | OK | Direct (temporal features) |
| `manager` | OK | Via sales_teams join |
| `regional_office` | OK | Via sales_teams join |
| `product_series` | OK | Via products join |
| `product_sales_price` | OK | Via products join |
| `account_sector` | OK | Via accounts join |
| `account_revenue` | OK | Via accounts join |

---

## Files

| File | Rows | Description |
|------|------|-------------|
| `sales_pipeline.csv` | 8,800 | Primary ML dataset (deal-level) |
| `accounts.csv` | 85 | Account metadata for feature enrichment |
| `products.csv` | 7 | Product metadata (series, price) |
| `sales_teams.csv` | 35 | Org structure (manager, region) |
| `data_dictionary.csv` | 23 | Schema documentation |

---

## Known Issues & Resolutions

| Issue | Impact | Resolution |
|-------|--------|------------|
| `GTXPro` vs `GTX Pro` | 1,480 deals can't join to products | Map in preprocessing |
| 16% NULL accounts | Missing account features | Use "Unknown" category |
| 5 agents in master not in pipeline | None | Informational only |

---

## Train/Test Split

| Set | Criteria | Deals | Win Rate |
|-----|----------|-------|----------|
| Train | `engage_date < 2017-07-01` | 3,649 | 65.6% |
| Test | `engage_date >= 2017-07-01` | 3,062 | 60.2% |

---


