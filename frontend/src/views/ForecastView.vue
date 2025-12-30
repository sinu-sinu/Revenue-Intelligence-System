<script setup lang="ts">
import { ref, computed } from 'vue'
import MetricCard from '@/components/common/MetricCard.vue'
import ForecastChart from '@/components/forecast/ForecastChart.vue'
import { useForecast } from '@/composables/useForecast'
import type { ForecastRequest } from '@/api/types'
import {
  ChartBarIcon,
  CheckBadgeIcon,
  ArrowTrendingUpIcon,
  DocumentTextIcon,
  CalendarDaysIcon,
} from '@heroicons/vue/24/solid'

const horizonWeeks = ref(12)

const request = computed<ForecastRequest>(() => ({
  horizon_weeks: horizonWeeks.value,
  period: 'week',
}))

const { data: forecast, isLoading, error } = useForecast(request)

function formatCurrency(value: number): string {
  if (value >= 1000000) {
    return `$${(value / 1000000).toFixed(1)}M`
  }
  if (value >= 1000) {
    return `$${(value / 1000).toFixed(0)}K`
  }
  return `$${value.toFixed(0)}`
}
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between animate-in">
      <div>
        <h1 class="text-3xl font-semibold text-surface-50 tracking-tight">Revenue Forecast</h1>
        <p class="mt-2 text-surface-400">Monte Carlo simulation with probability bands</p>
      </div>
      <div class="flex items-center gap-3">
        <CalendarDaysIcon class="w-5 h-5 text-surface-500 flex-shrink-0" />
        <select
          v-model.number="horizonWeeks"
          class="bg-surface-800 border border-surface-700 rounded-xl pl-4 pr-10 py-2.5 text-surface-200 text-sm focus:ring-2 focus:ring-accent-500 focus:border-accent-500 appearance-none bg-no-repeat bg-right"
          style="background-image: url('data:image/svg+xml;charset=UTF-8,%3csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 24 24%27 fill=%27none%27 stroke=%27%2394a3b8%27 stroke-width=%272%27%3e%3cpath d=%27M6 9l6 6 6-6%27/%3e%3c/svg%3e'); background-size: 1.25rem; background-position: right 0.75rem center;"
        >
          <option :value="4">4 weeks</option>
          <option :value="8">8 weeks</option>
          <option :value="12">12 weeks</option>
          <option :value="26">26 weeks</option>
        </select>
      </div>
    </div>

    <!-- Loading -->
    <div v-if="isLoading" class="flex items-center justify-center h-64">
      <div class="w-10 h-10 border-4 border-accent-500 rounded-full border-t-transparent animate-spin"></div>
    </div>

    <!-- Error -->
    <div v-else-if="error" class="card-dark p-6 border-rose-500/30">
      <h2 class="text-lg font-semibold text-rose-400">Error Loading Forecast</h2>
      <p class="text-surface-400 mt-2 text-sm">{{ error.message }}</p>
    </div>

    <!-- Forecast Content -->
    <template v-else-if="forecast">
      <!-- Summary Metrics -->
      <div class="grid grid-cols-1 md:grid-cols-4 gap-5 animate-in stagger-1">
        <MetricCard
          label="Total Pipeline"
          :value="forecast.summary.total_pipeline"
          :icon="ChartBarIcon"
          variant="accent"
        />
        <MetricCard
          label="Expected Revenue"
          :value="forecast.summary.expected_revenue"
          :icon="CheckBadgeIcon"
          variant="success"
        />
        <MetricCard
          label="P10-P90 Range"
          :value="`${formatCurrency(forecast.summary.p10_forecast)}-${formatCurrency(forecast.summary.p90_forecast)}`"
          :icon="ArrowTrendingUpIcon"
          variant="warning"
        />
        <MetricCard
          label="Active Deals"
          :value="forecast.summary.total_deals"
          :icon="DocumentTextIcon"
          variant="accent"
        />
      </div>

      <!-- Forecast Chart -->
      <div class="card-dark p-6 animate-in stagger-2">
        <h2 class="text-lg font-semibold text-surface-50 mb-4">Cumulative Revenue Forecast</h2>
        <ForecastChart :periods="forecast.periods" />
        <p class="text-xs text-surface-500 mt-4 text-center font-mono">
          Based on {{ forecast.summary.simulation_runs.toLocaleString() }} Monte Carlo simulations
        </p>
      </div>

      <!-- Forecast Breakdown -->
      <div class="card-dark p-6 animate-in stagger-3 overflow-hidden">
        <h2 class="text-lg font-semibold text-surface-50 mb-4">Period Breakdown</h2>
        <div class="overflow-x-auto">
          <table class="min-w-full divide-y divide-surface-800">
            <thead class="bg-surface-800/50">
              <tr>
                <th class="px-4 py-3 text-left text-xs font-semibold text-surface-400 uppercase">Period</th>
                <th class="px-4 py-3 text-right text-xs font-semibold text-surface-400 uppercase">P10 (Pessimistic)</th>
                <th class="px-4 py-3 text-right text-xs font-semibold text-surface-400 uppercase">P50 (Expected)</th>
                <th class="px-4 py-3 text-right text-xs font-semibold text-surface-400 uppercase">P90 (Optimistic)</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-surface-800/50">
              <tr v-for="period in forecast.periods" :key="period.period" class="hover:bg-surface-800/30 transition-colors">
                <td class="px-4 py-3 text-sm text-surface-200 font-medium">{{ period.period }}</td>
                <td class="px-4 py-3 text-sm text-rose-400 text-right font-mono">{{ formatCurrency(period.p10) }}</td>
                <td class="px-4 py-3 text-sm text-accent-400 text-right font-mono font-semibold">{{ formatCurrency(period.p50) }}</td>
                <td class="px-4 py-3 text-sm text-emerald-400 text-right font-mono">{{ formatCurrency(period.p90) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Top Contributing Deals -->
      <div class="card-dark p-6 animate-in stagger-4">
        <h2 class="text-lg font-semibold text-surface-50 mb-4">Top Contributing Deals</h2>
        <div class="space-y-3">
          <div
            v-for="deal in forecast.deal_forecasts.slice(0, 10)"
            :key="deal.opportunity_id"
            class="flex items-center justify-between p-4 bg-surface-800/30 rounded-xl hover:bg-surface-800/50 transition-colors border border-surface-800"
          >
            <div>
              <p class="text-surface-100 font-medium">{{ deal.account }}</p>
              <p class="text-sm text-surface-500">{{ deal.product }}</p>
            </div>
            <div class="text-right">
              <p class="text-accent-400 font-semibold font-mono">{{ formatCurrency(deal.expected_value) }}</p>
              <p class="text-sm text-surface-500 font-mono">{{ (deal.win_probability * 100).toFixed(0) }}% x {{ formatCurrency(deal.product_sales_price) }}</p>
            </div>
          </div>
        </div>
      </div>
    </template>
  </div>
</template>
