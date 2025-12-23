# Setup script for Revenue Intelligence System (PowerShell)

Write-Host "🚀 Setting up Revenue Intelligence System..." -ForegroundColor Green
Write-Host ""

# Check if Docker is running
try {
    docker info | Out-Null
    Write-Host "✓ Docker is running" -ForegroundColor Green
}
catch {
    Write-Host "❌ Docker is not running. Please start Docker first." -ForegroundColor Red
    exit 1
}

# Create .env file if it doesn't exist
if (-Not (Test-Path ".env")) {
    Write-Host "📝 Creating .env file..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host "✓ .env created (edit if needed)" -ForegroundColor Green
}

# Start Docker services
Write-Host ""
Write-Host "🐳 Starting Docker services..." -ForegroundColor Cyan
Set-Location docker
docker-compose up -d

# Wait for database to be ready
Write-Host ""
Write-Host "⏳ Waiting for database to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check database connection
Write-Host "🔍 Checking database connection..." -ForegroundColor Cyan
docker exec revenue_intel_db pg_isready -U app -d revenue_intel

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Database is ready" -ForegroundColor Green
}
else {
    Write-Host "❌ Database connection failed" -ForegroundColor Red
    exit 1
}

# Seed demo data
Write-Host ""
Write-Host "🌱 Seeding demo data..." -ForegroundColor Cyan
Set-Location ..
if (Test-Path "venv\Scripts\activate.ps1") {
    & "venv\Scripts\activate.ps1"
}
python database/seeds/seed_demo_data.py

Write-Host ""
Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Open http://localhost:8501 in your browser"
Write-Host "  2. Explore the Risk Dashboard"
Write-Host "  3. Start building Phase 1A (ML Pipeline)"
Write-Host ""
Write-Host "Useful commands:"
Write-Host "  - View logs: docker-compose logs -f"
Write-Host "  - Stop services: docker-compose down"
Write-Host "  - Database shell: docker exec -it revenue_intel_db psql -U app -d revenue_intel"

