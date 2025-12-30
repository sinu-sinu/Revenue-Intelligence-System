import { useQuery } from '@tanstack/vue-query'
import { computed, type Ref } from 'vue'
import { getDeals, getDeal, getSummary, getFilters } from '@/api/client'
import type { DealFilters } from '@/api/types'

export function useDeals(filters: Ref<DealFilters>) {
  return useQuery({
    queryKey: ['deals', filters],
    queryFn: () => getDeals(filters.value),
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}

export function useDeal(id: Ref<string>) {
  return useQuery({
    queryKey: ['deal', id],
    queryFn: () => getDeal(id.value),
    staleTime: 5 * 60 * 1000,
    enabled: computed(() => !!id.value),
  })
}

export function useSummary() {
  return useQuery({
    queryKey: ['summary'],
    queryFn: getSummary,
    staleTime: 5 * 60 * 1000,
  })
}

export function useFilters() {
  return useQuery({
    queryKey: ['filters'],
    queryFn: getFilters,
    staleTime: 10 * 60 * 1000, // 10 minutes
  })
}
