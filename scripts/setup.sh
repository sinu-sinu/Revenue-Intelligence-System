#!/bin/bash
# Setup script for Revenue Intelligence System

echo "🚀 Setting up Revenue Intelligence System..."
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

echo "✓ Docker is running"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "✓ .env created (edit if needed)"
fi

# Start Docker services
echo ""
echo "🐳 Starting Docker services..."
cd docker
docker-compose up -d

# Wait for database to be ready
echo ""
echo "⏳ Waiting for database to be ready..."
sleep 10

# Check database connection
echo "🔍 Checking database connection..."
docker exec revenue_intel_db pg_isready -U app -d revenue_intel

if [ $? -eq 0 ]; then
    echo "✓ Database is ready"
else
    echo "❌ Database connection failed"
    exit 1
fi

# Seed demo data
echo ""
echo "🌱 Seeding demo data..."
cd ..
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null
python database/seeds/seed_demo_data.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Open http://localhost:8501 in your browser"
echo "  2. Explore the Risk Dashboard"
echo "  3. Start building Phase 1A (ML Pipeline)"
echo ""
echo "Useful commands:"
echo "  - View logs: docker-compose logs -f"
echo "  - Stop services: docker-compose down"
echo "  - Database shell: docker exec -it revenue_intel_db psql -U app -d revenue_intel"

