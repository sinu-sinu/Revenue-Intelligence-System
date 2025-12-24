-- Quick seed data via SQL (simplified version)
-- This bypasses the Python connection issue

-- Insert Accounts
INSERT INTO accounts (id, name, industry, employee_count, annual_revenue, region) VALUES
(gen_random_uuid(), 'Acme Corporation', 'Technology', 500, 50000000, 'West'),
(gen_random_uuid(), 'TechStart Inc', 'Technology', 150, 15000000, 'East'),
(gen_random_uuid(), 'GlobalFin Solutions', 'Finance', 1200, 120000000, 'Central'),
(gen_random_uuid(), 'InnovateLabs', 'Healthcare', 300, 30000000, 'West'),
(gen_random_uuid(), 'Enterprise Co', 'Manufacturing', 2000, 200000000, 'Central'),
(gen_random_uuid(), 'NextGen Systems', 'Technology', 450, 45000000, 'East'),
(gen_random_uuid(), 'DataDriven Inc', 'Technology', 250, 25000000, 'West'),
(gen_random_uuid(), 'CloudScale Corp', 'Technology', 800, 80000000, 'Central'),
(gen_random_uuid(), 'AI Innovations', 'Technology', 180, 18000000, 'West'),
(gen_random_uuid(), 'Digital Dynamics', 'Retail', 600, 60000000, 'East'),
(gen_random_uuid(), 'FutureWorks', 'Technology', 220, 22000000, 'Central'),
(gen_random_uuid(), 'Synergy Solutions', 'Finance', 900, 90000000, 'West'),
(gen_random_uuid(), 'Quantum Computing', 'Technology', 120, 12000000, 'East'),
(gen_random_uuid(), 'Blockchain Ventures', 'Finance', 150, 15000000, 'West'),
(gen_random_uuid(), 'CyberSafe Systems', 'Technology', 350, 35000000, 'Central')
ON CONFLICT DO NOTHING;

-- Insert Sales Reps
INSERT INTO sales_reps (id, name, email, team, hire_date) VALUES
(gen_random_uuid(), 'Sarah Johnson', 'sarah.johnson@company.com', 'Enterprise', '2023-01-15'),
(gen_random_uuid(), 'Michael Chen', 'michael.chen@company.com', 'Mid-Market', '2022-06-20'),
(gen_random_uuid(), 'Emily Rodriguez', 'emily.rodriguez@company.com', 'SMB', '2023-03-10'),
(gen_random_uuid(), 'David Kim', 'david.kim@company.com', 'Enterprise', '2022-11-05'),
(gen_random_uuid(), 'Jessica Taylor', 'jessica.taylor@company.com', 'Mid-Market', '2023-02-28'),
(gen_random_uuid(), 'Robert Martinez', 'robert.martinez@company.com', 'SMB', '2022-09-15'),
(gen_random_uuid(), 'Amanda White', 'amanda.white@company.com', 'Enterprise', '2023-04-12'),
(gen_random_uuid(), 'Christopher Brown', 'christopher.brown@company.com', 'Mid-Market', '2022-12-01')
ON CONFLICT DO NOTHING;

-- Insert Products
INSERT INTO products (id, name, category, base_price) VALUES
(gen_random_uuid(), 'GTM Suite', 'Software', 50000),
(gen_random_uuid(), 'Analytics Platform', 'Software', 75000),
(gen_random_uuid(), 'Integration Hub', 'Software', 30000),
(gen_random_uuid(), 'Support Package', 'Service', 15000)
ON CONFLICT DO NOTHING;

-- Insert 50 Deals (using CTEs for random data)
WITH 
  account_ids AS (SELECT id FROM accounts ORDER BY random() LIMIT 50),
  rep_ids AS (SELECT id FROM sales_reps),
  product_ids AS (SELECT id FROM products),
  deal_data AS (
    SELECT 
      row_number() OVER () as rn,
      (SELECT id FROM account_ids OFFSET (random() * 14)::int LIMIT 1) as account_id,
      (SELECT id FROM rep_ids OFFSET (random() * 7)::int LIMIT 1) as rep_id,
      (SELECT id FROM product_ids OFFSET (random() * 3)::int LIMIT 1) as product_id,
      (10000 + random() * 490000)::int as amount,
      CASE (random() * 4)::int
        WHEN 0 THEN 'Prospecting'
        WHEN 1 THEN 'Qualification'
        WHEN 2 THEN 'Needs Analysis'
        WHEN 3 THEN 'Value Proposition'
        ELSE 'Negotiation'
      END as stage,
      NOW() - (random() * 90 || ' days')::interval as created_at,
      CASE WHEN random() < 0.3 THEN true ELSE false END as is_closed
    FROM generate_series(1, 50)
  )
INSERT INTO deals (id, name, account_id, owner_id, product_id, amount, stage, stage_entered_at, created_at, expected_close_date, actual_close_date, is_won, source, deal_type)
SELECT 
  gen_random_uuid(),
  'Deal ' || rn,
  account_id,
  rep_id,
  product_id,
  amount,
  CASE WHEN is_closed THEN 
    CASE WHEN random() < 0.6 THEN 'Closed Won' ELSE 'Closed Lost' END
  ELSE stage END,
  created_at + (random() * 10 || ' days')::interval,
  created_at,
  created_at + (random() * 60 + 30 || ' days')::interval,
  CASE WHEN is_closed THEN created_at + (random() * 50 + 20 || ' days')::interval ELSE NULL END,
  CASE WHEN is_closed THEN (random() < 0.6) ELSE NULL END,
  CASE (random() * 4)::int
    WHEN 0 THEN 'Inbound'
    WHEN 1 THEN 'Outbound'
    WHEN 2 THEN 'Referral'
    WHEN 3 THEN 'Partner'
    ELSE 'Event'
  END,
  CASE (random() * 2)::int
    WHEN 0 THEN 'New'
    WHEN 1 THEN 'Expansion'
    ELSE 'Renewal'
  END
FROM deal_data
ON CONFLICT DO NOTHING;

-- Show summary
SELECT 'Accounts' as table_name, COUNT(*) as count FROM accounts
UNION ALL
SELECT 'Sales Reps', COUNT(*) FROM sales_reps
UNION ALL
SELECT 'Products', COUNT(*) FROM products
UNION ALL
SELECT 'Deals', COUNT(*) FROM deals;

