import { useQuery } from '@tanstack/vue-query'
import { type Ref } from 'vue'
import { getForecast } from '@/api/client'
import type { ForecastRequest } from '@/api/types'

export function useForecast(request: Ref<ForecastRequest>) {
  return useQuery({
    queryKey: ['forecast', request],
    queryFn: () => getForecast(request.value),
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}
