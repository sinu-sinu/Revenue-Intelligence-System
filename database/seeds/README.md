# Database Seeds

This directory contains scripts to seed the database with demo data.

## Usage

### Via Docker

The seed script can be run after the database is initialized:

```bash
# Start the database
cd docker
docker-compose up db -d

# Wait for database to be ready
sleep 5

# Run seed script
docker exec -it revenue_intel_db python /docker-entrypoint-initdb.d/seeds/seed_demo_data.py
```

### Locally

If you have Python and PostgreSQL access locally:

```bash
# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run seed script
python database/seeds/seed_demo_data.py
```

## What Gets Created

- **15 Accounts** - Sample B2B companies across industries
- **8 Sales Reps** - Team members across Enterprise, Mid-Market, SMB
- **4 Products** - Mix of software and services
- **50 Deals** - Mix of open (70%) and closed (30%) opportunities
- **Stage History** - Historical stage transitions for velocity analysis

## Resetting Data

To reset and reseed:

```bash
# Connect to database
docker exec -it revenue_intel_db psql -U app -d revenue_intel

# Truncate tables (resets data, keeps schema)
TRUNCATE accounts, sales_reps, products, deals, deal_stage_history CASCADE;

# Exit and reseed
\q
python database/seeds/seed_demo_data.py
```

## For Production

**Important**: This seed data is for **development/demo only**. 

For production:
1. Load real CRM data via ETL pipeline
2. Implement proper data validation
3. Use migrations for schema changes

