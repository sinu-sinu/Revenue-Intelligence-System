<script setup lang="ts">
import { type Component } from 'vue'
import RiskBadge from '@/components/common/RiskBadge.vue'
import type { DealDetail } from '@/api/types'
import {
  ClockIcon,
  ArrowTrendingDownIcon,
  CurrencyDollarIcon,
  ExclamationTriangleIcon,
  MapPinIcon,
  BellAlertIcon,
  EyeIcon,
  CheckCircleIcon,
  UserIcon,
} from '@heroicons/vue/24/solid'

defineProps<{
  deal: DealDetail
}>()

function getActionRecommendation(deal: DealDetail): {
  action: string
  description: string
  color: string
  bgColor: string
  icon: Component
} {
  if (deal.risk_category === 'Critical' || deal.win_probability < 0.3) {
    return {
      action: 'Executive Call',
      description: 'Schedule immediate executive involvement',
      color: 'text-rose-400',
      bgColor: 'bg-rose-500/20',
      icon: BellAlertIcon,
    }
  }
  if (deal.risk_category === 'High') {
    return {
      action: 'Recovery Plan',
      description: 'Create detailed action plan to address risks',
      color: 'text-orange-400',
      bgColor: 'bg-orange-500/20',
      icon: ExclamationTriangleIcon,
    }
  }
  if (deal.risk_category === 'Medium') {
    return {
      action: 'Monitor Closely',
      description: 'Regular check-ins and progress tracking',
      color: 'text-amber-400',
      bgColor: 'bg-amber-500/20',
      icon: EyeIcon,
    }
  }
  return {
    action: 'Standard Process',
    description: 'Continue normal sales process',
    color: 'text-emerald-400',
    bgColor: 'bg-emerald-500/20',
    icon: CheckCircleIcon,
  }
}

function getDriverIcon(icon: string): Component {
  const iconMap: Record<string, Component> = {
    hourglass: ClockIcon,
    chart_decreasing: ArrowTrendingDownIcon,
    dollar: CurrencyDollarIcon,
    warning: ExclamationTriangleIcon,
  }
  return iconMap[icon] || MapPinIcon
}

function getDriverColor(icon: string): string {
  const colorMap: Record<string, string> = {
    hourglass: 'text-blue-400 bg-blue-500/20',
    chart_decreasing: 'text-rose-400 bg-rose-500/20',
    dollar: 'text-emerald-400 bg-emerald-500/20',
    warning: 'text-amber-400 bg-amber-500/20',
  }
  return colorMap[icon] || 'text-surface-400 bg-surface-700'
}
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="card-dark p-6 animate-in">
      <div class="flex items-start justify-between">
        <div>
          <h1 class="text-2xl font-semibold text-surface-50">{{ deal.account }}</h1>
          <p class="text-surface-400 mt-1">{{ deal.product }} - {{ deal.deal_stage }}</p>
          <p class="text-sm text-surface-500 mt-2 font-mono">ID: {{ deal.opportunity_id }}</p>
        </div>
        <RiskBadge :category="deal.risk_category" :score="deal.risk_score" size="lg" />
      </div>
    </div>

    <!-- Metrics Grid -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-4 animate-in stagger-1">
      <!-- Win Probability -->
      <div class="card-dark p-5">
        <p class="text-sm text-surface-400">Win Probability</p>
        <div class="mt-3">
          <div class="flex items-center justify-between mb-2">
            <span class="text-2xl font-semibold text-surface-50 font-mono">{{ (deal.win_probability * 100).toFixed(0) }}%</span>
          </div>
          <div class="w-full bg-surface-800 rounded-full h-2">
            <div
              class="bg-gradient-accent h-2 rounded-full transition-all duration-500"
              :style="{ width: `${deal.win_probability * 100}%` }"
            ></div>
          </div>
        </div>
      </div>

      <!-- Deal Value -->
      <div class="card-dark p-5">
        <p class="text-sm text-surface-400">Deal Value</p>
        <p class="text-2xl font-semibold text-surface-50 mt-3 font-mono">${{ deal.product_sales_price.toLocaleString() }}</p>
      </div>

      <!-- Predicted Close -->
      <div class="card-dark p-5">
        <p class="text-sm text-surface-400">Est. Close Days</p>
        <p class="text-2xl font-semibold text-surface-50 mt-3 font-mono">{{ deal.predicted_close_days }}</p>
        <p class="text-xs text-surface-500 mt-1">
          Range: {{ deal.predicted_close_range[0] }}-{{ deal.predicted_close_range[1] }} days
        </p>
      </div>

      <!-- Days in Stage -->
      <div class="card-dark p-5">
        <p class="text-sm text-surface-400">Days in Stage</p>
        <p class="text-2xl font-semibold text-surface-50 mt-3 font-mono">{{ deal.days_in_stage }}</p>
        <p class="text-xs text-surface-500 mt-1">{{ deal.deal_stage }}</p>
      </div>
    </div>

    <!-- Risk Drivers -->
    <div class="card-dark p-6 animate-in stagger-2">
      <h2 class="text-lg font-semibold text-surface-50 mb-4">Why This Risk Level</h2>
      <div v-if="deal.risk_drivers.length > 0" class="space-y-3">
        <div
          v-for="(driver, index) in deal.risk_drivers"
          :key="index"
          class="flex items-start gap-4 p-4 bg-surface-800/30 rounded-xl border border-surface-800"
        >
          <div :class="['w-10 h-10 rounded-xl flex items-center justify-center', getDriverColor(driver.icon)]">
            <component :is="getDriverIcon(driver.icon)" class="w-5 h-5" />
          </div>
          <div class="flex-1">
            <div class="flex items-center justify-between">
              <h3 class="font-medium text-surface-100">{{ driver.driver }}</h3>
              <span class="text-sm font-mono text-orange-400 bg-orange-500/20 px-2 py-0.5 rounded">{{ driver.impact }}</span>
            </div>
            <p class="text-sm text-surface-400 mt-1">{{ driver.detail }}</p>
          </div>
        </div>
      </div>
      <p v-else class="text-surface-500">No significant risk drivers identified.</p>
    </div>

    <!-- Recommended Action -->
    <div class="card-dark p-6 animate-in stagger-3">
      <h2 class="text-lg font-semibold text-surface-50 mb-4">Recommended Action</h2>
      <div class="flex items-center gap-4">
        <div
          :class="[
            'w-14 h-14 rounded-xl flex items-center justify-center',
            getActionRecommendation(deal).bgColor
          ]"
        >
          <component
            :is="getActionRecommendation(deal).icon"
            :class="['w-7 h-7', getActionRecommendation(deal).color]"
          />
        </div>
        <div>
          <p :class="['text-xl font-semibold', getActionRecommendation(deal).color]">
            {{ getActionRecommendation(deal).action }}
          </p>
          <p class="text-surface-400">{{ getActionRecommendation(deal).description }}</p>
        </div>
      </div>
    </div>

    <!-- Sales Rep Info -->
    <div class="card-dark p-6 animate-in stagger-4">
      <h2 class="text-lg font-semibold text-surface-50 mb-4">Sales Representative</h2>
      <div class="flex items-center gap-3">
        <div class="w-10 h-10 rounded-full bg-accent-500/20 flex items-center justify-center">
          <UserIcon class="w-5 h-5 text-accent-400" />
        </div>
        <p class="text-surface-100 font-medium">{{ deal.sales_agent }}</p>
      </div>
    </div>
  </div>
</template>
