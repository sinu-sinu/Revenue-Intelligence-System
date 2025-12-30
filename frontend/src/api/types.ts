// Deal types
export interface RiskDriver {
  driver: string
  detail: string
  impact: string
  icon: string
}

export interface DealSummary {
  opportunity_id: string
  account: string
  sales_agent: string
  product: string
  deal_stage: string
  win_probability: number
  risk_score: number
  risk_category: 'Low' | 'Medium' | 'High' | 'Critical'
  product_sales_price: number
}

export interface DealDetail extends DealSummary {
  predicted_close_days: number
  predicted_close_range: [number, number]
  days_in_stage: number
  risk_drivers: RiskDriver[]
}

export interface DealFilters {
  account?: string[]
  sales_agent?: string[]
  product?: string[]
  risk_category?: string[]
  min_risk_score?: number
  max_risk_score?: number
  sort_by?: string
  sort_order?: 'asc' | 'desc'
  limit?: number
}

export interface DealListResponse {
  deals: DealSummary[]
  total: number
  limit: number
}

export interface SummaryStats {
  total_deals: number
  at_risk_revenue: number
  high_risk_count: number
  avg_win_probability: number
  risk_distribution: Record<string, number>
}

export interface FilterOptions {
  accounts: string[]
  sales_agents: string[]
  products: string[]
  risk_categories: string[]
}

// Forecast types
export interface ForecastRequest {
  horizon_weeks?: number
  period?: 'week' | 'month'
  accounts?: string[]
  sales_agents?: string[]
  products?: string[]
}

export interface ForecastPeriod {
  period: string
  p10: number
  p50: number
  p90: number
}

export interface DealForecast {
  opportunity_id: string
  account: string
  product: string
  win_probability: number
  product_sales_price: number
  expected_value: number
  expected_close_days: number
}

export interface ForecastSummary {
  forecast_date: string
  horizon_days: number
  total_deals: number
  total_pipeline: number
  expected_revenue: number
  p10_forecast: number
  p50_forecast: number
  p90_forecast: number
  simulation_runs: number
}

export interface ForecastResponse {
  periods: ForecastPeriod[]
  deal_forecasts: DealForecast[]
  summary: ForecastSummary
}
