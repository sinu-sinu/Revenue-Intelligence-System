import { ref, watch, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import type { DealFilters } from '@/api/types'

export function useFilterState() {
  const route = useRoute()
  const router = useRouter()

  // Initialize from URL query params
  const account = ref<string[]>(
    route.query.account
      ? Array.isArray(route.query.account)
        ? (route.query.account as string[])
        : [route.query.account as string]
      : []
  )
  const salesAgent = ref<string[]>(
    route.query.sales_agent
      ? Array.isArray(route.query.sales_agent)
        ? (route.query.sales_agent as string[])
        : [route.query.sales_agent as string]
      : []
  )
  const product = ref<string[]>(
    route.query.product
      ? Array.isArray(route.query.product)
        ? (route.query.product as string[])
        : [route.query.product as string]
      : []
  )
  const riskCategory = ref<string[]>(
    route.query.risk_category
      ? Array.isArray(route.query.risk_category)
        ? (route.query.risk_category as string[])
        : [route.query.risk_category as string]
      : []
  )
  const minRiskScore = ref(Number(route.query.min_risk_score) || 0)
  const maxRiskScore = ref(Number(route.query.max_risk_score) || 100)
  const sortBy = ref((route.query.sort_by as string) || 'risk_score')
  const sortOrder = ref<'asc' | 'desc'>((route.query.sort_order as 'asc' | 'desc') || 'desc')

  // Computed filters object
  const filters = computed<DealFilters>(() => ({
    account: account.value.length ? account.value : undefined,
    sales_agent: salesAgent.value.length ? salesAgent.value : undefined,
    product: product.value.length ? product.value : undefined,
    risk_category: riskCategory.value.length ? riskCategory.value : undefined,
    min_risk_score: minRiskScore.value,
    max_risk_score: maxRiskScore.value,
    sort_by: sortBy.value,
    sort_order: sortOrder.value,
  }))

  // Sync filters to URL
  watch(
    filters,
    (newFilters) => {
      const query: Record<string, string | string[]> = {}

      if (newFilters.account?.length) query.account = newFilters.account
      if (newFilters.sales_agent?.length) query.sales_agent = newFilters.sales_agent
      if (newFilters.product?.length) query.product = newFilters.product
      if (newFilters.risk_category?.length) query.risk_category = newFilters.risk_category
      if (newFilters.min_risk_score) query.min_risk_score = String(newFilters.min_risk_score)
      if (newFilters.max_risk_score !== 100) query.max_risk_score = String(newFilters.max_risk_score)
      if (newFilters.sort_by && newFilters.sort_by !== 'risk_score') query.sort_by = newFilters.sort_by
      if (newFilters.sort_order && newFilters.sort_order !== 'desc') query.sort_order = newFilters.sort_order

      router.replace({ query })
    },
    { deep: true }
  )

  // Reset filters
  function resetFilters() {
    account.value = []
    salesAgent.value = []
    product.value = []
    riskCategory.value = []
    minRiskScore.value = 0
    maxRiskScore.value = 100
    sortBy.value = 'risk_score'
    sortOrder.value = 'desc'
  }

  return {
    account,
    salesAgent,
    product,
    riskCategory,
    minRiskScore,
    maxRiskScore,
    sortBy,
    sortOrder,
    filters,
    resetFilters,
  }
}
