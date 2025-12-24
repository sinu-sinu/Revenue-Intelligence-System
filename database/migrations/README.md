# Database Migrations

This directory contains database migration scripts for the Revenue Intelligence System.

## Structure

- `schema.sql` - Main schema in parent directory
- This folder can contain versioned migrations if needed

## Running Migrations

The schema is automatically applied when the database container starts via Docker Compose.

For manual migrations:

```bash
# Connect to database
docker exec -it revenue_intel_db psql -U app -d revenue_intel

# Or from host (if psql installed)
psql postgresql://app:dev_password@localhost:5433/revenue_intel -f database/schema.sql
```

## Future Migration Pattern

If you need to add migrations:

```
migrations/
  001_initial_schema.sql
  002_add_deal_notes.sql
  003_add_activity_tracking.sql
```

Consider using a migration tool like Alembic or Flyway for production.

