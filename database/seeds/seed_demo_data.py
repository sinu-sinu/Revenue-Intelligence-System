"""
Seed demo data for Revenue Intelligence System.

This script creates realistic sample data for development and demos.
"""

import random
from datetime import datetime, timedelta
from uuid import uuid4

import psycopg2
from psycopg2.extras import execute_values

# Database connection
DATABASE_URL = "postgresql://app:dev_password@localhost:5433/revenue_intel"

# Sample data
INDUSTRIES = ["Technology", "Healthcare", "Finance", "Manufacturing", "Retail"]
REGIONS = ["West", "East", "Central", "North", "South"]
TEAMS = ["Enterprise", "Mid-Market", "SMB"]
STAGES = [
    "Prospecting",
    "Qualification",
    "Needs Analysis",
    "Value Proposition",
    "Negotiation",
]
SOURCES = ["Inbound", "Outbound", "Referral", "Partner", "Event"]
DEAL_TYPES = ["New", "Expansion", "Renewal"]
PRODUCTS = [
    ("GTM Suite", "Software", 50000),
    ("Analytics Platform", "Software", 75000),
    ("Integration Hub", "Software", 30000),
    ("Support Package", "Service", 15000),
]

# Sample account names
ACCOUNT_NAMES = [
    "Acme Corporation",
    "TechStart Inc",
    "GlobalFin Solutions",
    "InnovateLabs",
    "Enterprise Co",
    "NextGen Systems",
    "DataDriven Inc",
    "CloudScale Corp",
    "AI Innovations",
    "Digital Dynamics",
    "FutureWorks",
    "Synergy Solutions",
    "Quantum Computing",
    "Blockchain Ventures",
    "CyberSafe Systems",
]

# Sample rep names
REP_NAMES = [
    "Sarah Johnson",
    "Michael Chen",
    "Emily Rodriguez",
    "David Kim",
    "Jessica Taylor",
    "Robert Martinez",
    "Amanda White",
    "Christopher Brown",
]


def create_demo_data():
    """Generate and insert demo data."""

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    print("Creating demo data...")

    # 1. Create Accounts
    accounts = []
    for name in ACCOUNT_NAMES:
        accounts.append(
            (
                str(uuid4()),
                name,
                random.choice(INDUSTRIES),
                random.randint(50, 5000),
                random.randint(1_000_000, 100_000_000),
                random.choice(REGIONS),
            )
        )

    execute_values(
        cur,
        """
        INSERT INTO accounts (id, name, industry, employee_count, annual_revenue, region)
        VALUES %s
        ON CONFLICT DO NOTHING
    """,
        accounts,
    )
    print(f"[OK] Created {len(accounts)} accounts")

    # 2. Create Sales Reps
    reps = []
    for name in REP_NAMES:
        reps.append(
            (
                str(uuid4()),
                name,
                f"{name.lower().replace(' ', '.')}@company.com",
                random.choice(TEAMS),
                datetime.now() - timedelta(days=random.randint(100, 1000)),
            )
        )

    execute_values(
        cur,
        """
        INSERT INTO sales_reps (id, name, email, team, hire_date)
        VALUES %s
        ON CONFLICT DO NOTHING
    """,
        reps,
    )
    print(f"[OK] Created {len(reps)} sales reps")

    # 3. Create Products
    products = []
    for name, category, base_price in PRODUCTS:
        products.append((str(uuid4()), name, category, base_price))

    execute_values(
        cur,
        """
        INSERT INTO products (id, name, category, base_price)
        VALUES %s
        ON CONFLICT DO NOTHING
    """,
        products,
    )
    print(f"[OK] Created {len(products)} products")

    # Get IDs for foreign keys
    cur.execute("SELECT id FROM accounts")
    account_ids = [row[0] for row in cur.fetchall()]

    cur.execute("SELECT id FROM sales_reps")
    rep_ids = [row[0] for row in cur.fetchall()]

    cur.execute("SELECT id FROM products")
    product_ids = [row[0] for row in cur.fetchall()]

    # 4. Create Deals (mix of open and closed)
    deals = []
    num_deals = 50

    for i in range(num_deals):
        deal_id = str(uuid4())
        account_id = random.choice(account_ids)
        owner_id = random.choice(rep_ids)
        product_id = random.choice(product_ids)
        amount = random.randint(10000, 500000)
        stage = random.choice(STAGES)
        created_at = datetime.now() - timedelta(days=random.randint(1, 90))
        stage_entered_at = created_at + timedelta(days=random.randint(1, 20))

        # 30% of deals are closed
        is_closed = random.random() < 0.3
        is_won = None
        actual_close_date = None

        if is_closed:
            is_won = random.random() < 0.6  # 60% win rate
            actual_close_date = created_at + timedelta(days=random.randint(20, 60))
            stage = "Closed Won" if is_won else "Closed Lost"

        deals.append(
            (
                deal_id,
                f"Deal {i+1}",
                account_id,
                owner_id,
                product_id,
                amount,
                stage,
                stage_entered_at,
                created_at,
                created_at + timedelta(days=random.randint(30, 90)),  # expected close
                actual_close_date,
                is_won,
                random.choice(SOURCES),
                random.choice(DEAL_TYPES),
            )
        )

    execute_values(
        cur,
        """
        INSERT INTO deals (
            id, name, account_id, owner_id, product_id, amount, stage,
            stage_entered_at, created_at, expected_close_date, actual_close_date,
            is_won, source, deal_type
        )
        VALUES %s
        ON CONFLICT DO NOTHING
    """,
        deals,
    )
    print(f"[OK] Created {len(deals)} deals")

    # 5. Create some stage history for realism
    stage_history = []
    for deal in random.sample(deals, min(20, len(deals))):
        deal_id = deal[0]
        # Add 2-3 historical stages
        for j in range(random.randint(2, 3)):
            stage_history.append(
                (
                    str(uuid4()),
                    deal_id,
                    random.choice(STAGES),
                    datetime.now() - timedelta(days=random.randint(10, 50)),
                    datetime.now() - timedelta(days=random.randint(1, 9)),
                )
            )

    if stage_history:
        execute_values(
            cur,
            """
            INSERT INTO deal_stage_history (id, deal_id, stage, entered_at, exited_at)
            VALUES %s
            ON CONFLICT DO NOTHING
        """,
            stage_history,
        )
        print(f"[OK] Created {len(stage_history)} stage history entries")

    conn.commit()
    cur.close()
    conn.close()

    print("\n[SUCCESS] Demo data seeded successfully!")
    print("\nQuick Stats:")
    print(f"  - {len(accounts)} Accounts")
    print(f"  - {len(reps)} Sales Reps")
    print(f"  - {len(products)} Products")
    print(f"  - {len(deals)} Deals")
    print(f"\nRun: docker exec -it revenue_intel_db psql -U app -d revenue_intel")


if __name__ == "__main__":
    create_demo_data()

