<script setup lang="ts">
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import MetricCard from '@/components/common/MetricCard.vue'
import { useSummary } from '@/composables/useDeals'
import {
  CurrencyDollarIcon,
  ExclamationTriangleIcon,
  ArrowTrendingUpIcon,
  ChartBarSquareIcon,
  ShieldExclamationIcon,
  ChartPieIcon,
} from '@heroicons/vue/24/solid'

const router = useRouter()
const { data: summary, isLoading } = useSummary()

function navigateTo(path: string) {
  router.push(path)
}

// Always show all 4 risk categories in order
const riskCategories = ['Critical', 'High', 'Medium', 'Low'] as const

const riskConfig = {
  Critical: { color: 'bg-rose-500', textColor: 'text-rose-400', bgColor: 'bg-rose-500/10', borderColor: 'border-rose-500/30' },
  High: { color: 'bg-orange-500', textColor: 'text-orange-400', bgColor: 'bg-orange-500/10', borderColor: 'border-orange-500/30' },
  Medium: { color: 'bg-amber-500', textColor: 'text-amber-400', bgColor: 'bg-amber-500/10', borderColor: 'border-amber-500/30' },
  Low: { color: 'bg-emerald-500', textColor: 'text-emerald-400', bgColor: 'bg-emerald-500/10', borderColor: 'border-emerald-500/30' },
}

const riskDistribution = computed(() => {
  if (!summary.value?.risk_distribution) return []
  return riskCategories.map(cat => ({
    category: cat,
    count: summary.value?.risk_distribution[cat] ?? 0,
    ...riskConfig[cat]
  }))
})
</script>

<template>
  <div class="space-y-8">
    <!-- Header -->
    <div class="animate-in">
      <h1 class="text-3xl font-semibold text-surface-50 tracking-tight">Dashboard</h1>
      <p class="mt-2 text-surface-400">Overview of your sales pipeline and risk assessment</p>
    </div>

    <!-- Loading State -->
    <div v-if="isLoading" class="flex items-center justify-center h-64">
      <div class="w-10 h-10 border-4 border-accent-500 rounded-full border-t-transparent animate-spin"></div>
    </div>

    <!-- Dashboard Content -->
    <template v-else-if="summary">
      <!-- Key Metrics -->
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5 animate-in stagger-1">
        <MetricCard
          label="At-Risk Revenue"
          :value="summary.at_risk_revenue"
          :icon="CurrencyDollarIcon"
          variant="danger"
        />
        <MetricCard
          label="High Risk Deals"
          :value="summary.high_risk_count"
          :icon="ExclamationTriangleIcon"
          variant="warning"
        />
        <MetricCard
          label="Avg Win Probability"
          :value="summary.avg_win_probability"
          :icon="ArrowTrendingUpIcon"
          variant="success"
        />
        <MetricCard
          label="Total Active Deals"
          :value="summary.total_deals"
          :icon="ChartBarSquareIcon"
          variant="accent"
        />
      </div>

      <!-- Risk Distribution -->
      <div class="card-dark p-6 animate-in stagger-2">
        <h2 class="text-lg font-semibold text-surface-50 mb-5">Risk Distribution</h2>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div
            v-for="item in riskDistribution"
            :key="item.category"
            :class="[
              'relative p-5 rounded-xl border transition-all duration-300 hover:scale-[1.02] cursor-default',
              item.bgColor,
              item.borderColor,
            ]"
          >
            <!-- Glow effect -->
            
            <p :class="['text-4xl font-bold font-mono tracking-tight', item.textColor]">
              {{ item.count }}
            </p>
            <div class="flex items-center gap-2 mt-3">
              <span :class="['w-2 h-2 rounded-full', item.color]" />
              <span class="text-sm text-surface-300 font-medium">{{ item.category }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Quick Actions -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-5 animate-in stagger-3">
        <button
          @click="navigateTo('/risk')"
          class="group card-dark p-6 text-left hover:shadow-card-hover transition-all duration-300 glow-border"
        >
          <div class="flex items-center gap-5">
            <div class="w-14 h-14 rounded-2xl bg-gradient-accent flex items-center justify-center shadow-glow group-hover:shadow-glow transition-shadow">
              <ShieldExclamationIcon class="w-7 h-7 text-surface-950" />
            </div>
            <div>
              <h3 class="text-lg font-semibold text-surface-50">Risk Dashboard</h3>
              <p class="text-sm text-surface-400 mt-1">View and filter at-risk deals</p>
            </div>
          </div>
        </button>

        <button
          @click="navigateTo('/forecast')"
          class="group card-dark p-6 text-left hover:shadow-card-hover transition-all duration-300 glow-border"
        >
          <div class="flex items-center gap-5">
            <div class="w-14 h-14 rounded-2xl bg-gradient-success flex items-center justify-center">
              <ChartPieIcon class="w-7 h-7 text-surface-950" />
            </div>
            <div>
              <h3 class="text-lg font-semibold text-surface-50">Revenue Forecast</h3>
              <p class="text-sm text-surface-400 mt-1">Monte Carlo simulation with confidence bands</p>
            </div>
          </div>
        </button>
      </div>
    </template>
  </div>
</template>
