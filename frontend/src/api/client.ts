import axios from 'axios'
import type {
  DealDetail,
  DealExplanation,
  DealFilters,
  DealListResponse,
  FilterOptions,
  ForecastRequest,
  ForecastResponse,
  SummaryStats,
} from './types'

const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
})

// Deals API
export async function getDeals(filters: DealFilters = {}): Promise<DealListResponse> {
  const params = new URLSearchParams()

  if (filters.account?.length) {
    filters.account.forEach((a) => params.append('account', a))
  }
  if (filters.sales_agent?.length) {
    filters.sales_agent.forEach((a) => params.append('sales_agent', a))
  }
  if (filters.product?.length) {
    filters.product.forEach((p) => params.append('product', p))
  }
  if (filters.risk_category?.length) {
    filters.risk_category.forEach((r) => params.append('risk_category', r))
  }
  if (filters.min_risk_score !== undefined) {
    params.set('min_risk_score', String(filters.min_risk_score))
  }
  if (filters.max_risk_score !== undefined) {
    params.set('max_risk_score', String(filters.max_risk_score))
  }
  if (filters.sort_by) {
    params.set('sort_by', filters.sort_by)
  }
  if (filters.sort_order) {
    params.set('sort_order', filters.sort_order)
  }
  if (filters.limit) {
    params.set('limit', String(filters.limit))
  }

  const { data } = await api.get<DealListResponse>(`/deals?${params.toString()}`)
  return data
}

export async function getDeal(id: string): Promise<DealDetail> {
  const { data } = await api.get<DealDetail>(`/deals/${id}`)
  return data
}

export async function getDealExplanation(id: string): Promise<DealExplanation> {
  const { data } = await api.get<DealExplanation>(`/deals/${id}/explanation`)
  return data
}

export async function getSummary(): Promise<SummaryStats> {
  const { data } = await api.get<SummaryStats>('/deals/summary')
  return data
}

export async function getFilters(): Promise<FilterOptions> {
  const { data } = await api.get<FilterOptions>('/deals/filters')
  return data
}

// Forecast API
export async function getForecast(request: ForecastRequest = {}): Promise<ForecastResponse> {
  const { data } = await api.post<ForecastResponse>('/forecast', request)
  return data
}

export default api
