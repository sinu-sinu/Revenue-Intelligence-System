# Dataset Backup - REJECTED FOR PHASE 1A

> **Status**: Archived, not suitable for ML use  
> **Review Date**: December 2024  
> **Decision**: Use `/dataset/sales_pipeline.csv` instead

---

## Why This Dataset Was Rejected

### Critical Issues

1. **Data Leakage**: 100% of open deals (Negotiating/Proposing) have `Closed Date` populated. This would cause severe target leakage if used for ML training.

2. **Missing Key Dimensions**: No `sales_agent`, `account`, or standardized `product` fields. Cannot build meaningful win-probability model without attribution dimensions.

3. **Wrong Grain**: Data is at line-item level (9,994 rows for 5,009 opportunities). Phase 1 requires deal-level predictions.

4. **Temporal Corruption**: 42 records have `Closed Date` before `Created Date` (physically impossible).

5. **Cannot Join to Reference Tables**: Different time period (2020-2024) and missing join keys make enrichment impossible.

---

## File Contents

| File | Rows | Description |
|------|------|-------------|
| `Sales Pipeline_backup.csv` | 9,994 | Line-item level sales data, 2020-2024 |

## Columns

| Column | Description | Issue |
|--------|-------------|-------|
| index | Row index | N/A |
| Row ID | Alternative row identifier | N/A |
| Opportunity ID | Deal identifier | Valid but multi-row per deal |
| Created Date | Deal creation date | Valid |
| Closed Date | Deal close date | **CORRUPTED**: Present on open deals |
| Stage | Deal stage | Different semantics than current |
| Opportunity Name | Product description | Free-text, not standardized |
| Sales | Revenue amount | **LEAKED**: Present on open deals |

---

## Cant be Use For

- ❌ Win probability modeling
- ❌ Time-to-close modeling
- ❌ Revenue forecasting
- ❌ Rep/Account performance analysis
- ❌ Any ML training

## cant Be Suitable For

- ⚠️ Data archaeology / audit trail
- ⚠️ Understanding historical data sources
- ⚠️ Comparison with other systems



