# Phase 2: Vue.js Frontend Refactor (Optional)

> **Duration**: ~5 days  
> **Trigger**: When you want to demonstrate frontend separation skills  
> **Goal**: Extract API, build modern Vue 3 frontend

---

## When to Do Phase 2

Consider this phase when:
- ✅ Phase 1 is complete and polished
- ✅ You want to show full-stack capabilities
- ✅ You have time to do it well (not rushed)
- ✅ You want role-based access patterns
- ✅ Interview requires frontend demonstration

Skip if:
- ❌ Phase 1 isn't fully polished
- ❌ Time is limited
- ❌ Target role is ML-focused, not full-stack

---

## Checklist

### 2.1 FastAPI Extraction
- [ ] Create `api/` directory structure:
  ```
  api/
  ├── __init__.py
  ├── main.py
  ├── routes/
  │   ├── deals.py
  │   ├── forecasts.py
  │   └── health.py
  ├── schemas/
  │   ├── deal.py
  │   └── forecast.py
  └── dependencies.py
  ```
- [ ] Define API schemas with Pydantic:
  ```python
  class DealResponse(BaseModel):
      id: str
      name: str
      account_name: str
      amount: float
      stage: str
      risk_score: int
      win_probability: float
      risk_drivers: list[RiskDriver]
      
  class RiskDriver(BaseModel):
      feature: str
      description: str
      impact: float
  ```
- [ ] Implement endpoints:
  ```python
  @router.get("/deals", response_model=list[DealSummary])
  async def list_deals(
      team: str | None = None,
      product: str | None = None,
      min_amount: float = 0,
      stage: str | None = None,
      sort_by: str = "risk_value"
  ):
      ...
  
  @router.get("/deals/{deal_id}", response_model=DealDetail)
  async def get_deal(deal_id: str):
      ...
  
  @router.get("/forecast", response_model=ForecastResponse)
  async def get_forecast(
      horizon_weeks: int = 12,
      team: str | None = None
  ):
      ...
  ```
- [ ] Add OpenAPI documentation (auto-generated)
- [ ] Implement CORS for local development

### 2.2 Vue 3 Project Setup
- [ ] Create Vue project with Vite:
  ```bash
  npm create vue@latest frontend -- --typescript
  cd frontend
  npm install
  ```
- [ ] Install dependencies:
  ```bash
  npm install @tanstack/vue-query axios
  npm install -D tailwindcss postcss autoprefixer
  ```
- [ ] Configure Tailwind with custom theme:
  ```javascript
  // tailwind.config.js
  export default {
    theme: {
      extend: {
        colors: {
          primary: {
            50: '#EEF2FF',
            500: '#6366F1',
            900: '#312E81',
          },
          slate: {
            850: '#172033',
          }
        }
      }
    }
  }
  ```

### 2.3 Vue Components
- [ ] Create component structure:
  ```
  frontend/src/
  ├── components/
  │   ├── common/
  │   │   ├── MetricCard.vue
  │   │   ├── RiskBadge.vue
  │   │   ├── ProbabilityGauge.vue
  │   │   └── DataTable.vue
  │   ├── deals/
  │   │   ├── DealList.vue
  │   │   ├── DealCard.vue
  │   │   └── DealDetail.vue
  │   └── forecast/
  │       ├── ForecastChart.vue
  │       └── ForecastTable.vue
  ├── views/
  │   ├── RiskDashboard.vue
  │   ├── DealView.vue
  │   └── ForecastView.vue
  ├── composables/
  │   ├── useDeals.ts
  │   └── useForecast.ts
  └── api/
      └── client.ts
  ```

- [ ] Implement data fetching with Vue Query:
  ```typescript
  // composables/useDeals.ts
  export function useDeals(filters: Ref<DealFilters>) {
    return useQuery({
      queryKey: ['deals', filters],
      queryFn: () => api.getDeals(filters.value),
      staleTime: 5 * 60 * 1000,
    })
  }
  ```

- [ ] Create RiskBadge component:
  ```vue
  <template>
    <span :class="badgeClass" class="px-2 py-1 rounded-full text-sm font-medium">
      {{ score }}
    </span>
  </template>
  
  <script setup lang="ts">
  const props = defineProps<{ score: number }>()
  
  const badgeClass = computed(() => {
    if (props.score >= 70) return 'bg-red-100 text-red-800'
    if (props.score >= 40) return 'bg-amber-100 text-amber-800'
    return 'bg-green-100 text-green-800'
  })
  </script>
  ```

### 2.4 Data Visualization
- [ ] Install and configure Chart.js or Apache ECharts:
  ```bash
  npm install echarts vue-echarts
  ```
- [ ] Create ForecastChart with uncertainty bands
- [ ] Add interactive tooltips
- [ ] Implement drill-down click handlers
- [ ] Create responsive layouts

### 2.5 State Management
- [ ] Use Vue 3 composables (no Vuex/Pinia needed for this size)
- [ ] Create filter state composable:
  ```typescript
  // composables/useFilters.ts
  export function useFilters() {
    const team = ref<string | null>(null)
    const product = ref<string | null>(null)
    const minAmount = ref(0)
    
    const filters = computed(() => ({
      team: team.value,
      product: product.value,
      min_amount: minAmount.value,
    }))
    
    return { team, product, minAmount, filters }
  }
  ```
- [ ] Persist filters in URL query params

### 2.6 Docker Integration
- [ ] Update `docker-compose.yml`:
  ```yaml
  services:
    api:
      build: 
        context: .
        dockerfile: docker/Dockerfile.api
      ports:
        - "8000:8000"
      depends_on:
        - db
    
    frontend:
      build:
        context: ./frontend
        dockerfile: Dockerfile
      ports:
        - "3000:80"
      depends_on:
        - api
    
    db:
      image: postgres:15-alpine
      # ...
  ```
- [ ] Create Nginx config for frontend:
  ```nginx
  server {
    listen 80;
    root /usr/share/nginx/html;
    
    location / {
      try_files $uri $uri/ /index.html;
    }
    
    location /api {
      proxy_pass http://api:8000;
    }
  }
  ```

### 2.7 Polish & Testing
- [ ] Add loading skeletons
- [ ] Implement error states
- [ ] Add unit tests for composables
- [ ] E2E tests with Playwright
- [ ] Ensure accessibility (ARIA labels, keyboard nav)

---

## Architecture After Phase 2

```
┌─────────────────────────────────────────────────────────────┐
│                      NGINX (Frontend)                        │
│                     http://localhost:3000                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                   Vue 3 + Vite                       │    │
│  │  • Vue Query for data fetching                      │    │
│  │  • TailwindCSS for styling                          │    │
│  │  • ECharts for visualizations                       │    │
│  └───────────────────────┬─────────────────────────────┘    │
└──────────────────────────┼──────────────────────────────────┘
                           │ REST API
┌──────────────────────────▼──────────────────────────────────┐
│                     FastAPI Backend                          │
│                    http://localhost:8000                     │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐    │
│  │ /deals      │  │ /forecast    │  │ /health          │    │
│  └─────────────┘  └──────────────┘  └──────────────────┘    │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   Core Business Logic                        │
│           (Same as Phase 1 - no rewrite!)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## What Changes vs What Stays

| Component | Status | Notes |
|-----------|--------|-------|
| ML Models | ✅ Same | No changes |
| Scoring Logic | ✅ Same | Extracted to service |
| Database | ✅ Same | No schema changes |
| Explanations | ✅ Same | API exposes same data |
| UI | 🔄 Replaced | Streamlit → Vue |
| API | 🆕 New | FastAPI layer added |

---

## Acceptance Criteria

✅ API documentation auto-generated (OpenAPI)  
✅ Vue app shows same data as Streamlit version  
✅ All visualizations working  
✅ Filters persist in URL  
✅ Docker Compose starts all services  
✅ No business logic duplicated in frontend  

---

## Portfolio Framing

When discussing the refactor:

> "I intentionally started with Streamlit to validate the ML logic and user experience quickly. Once that was solid, I extracted a proper API and built a Vue frontend. The core scoring and forecasting logic required zero changes — that's the benefit of good architecture."

This demonstrates:
1. Pragmatic decision-making (right tool for the phase)
2. Clean architecture (separation of concerns)
3. Full-stack capability
4. Refactoring skills (not rewriting)

