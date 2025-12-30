<script setup lang="ts">
import { computed, type Component } from 'vue'
import { ArrowTrendingUpIcon, ArrowTrendingDownIcon, MinusIcon } from '@heroicons/vue/24/solid'

const props = defineProps<{
  label: string
  value: string | number
  subValue?: string
  trend?: 'up' | 'down' | 'neutral'
  icon?: Component
  variant?: 'default' | 'accent' | 'danger' | 'warning' | 'success'
  progress?: number
}>()

function formatValue(value: string | number): string {
  if (typeof value === 'number') {
    if (value >= 1000000) {
      return `$${(value / 1000000).toFixed(1)}M`
    }
    if (value >= 1000) {
      return `$${(value / 1000).toFixed(0)}K`
    }
    if (value < 1 && value > 0) {
      return `${(value * 100).toFixed(0)}%`
    }
    return value.toLocaleString()
  }
  return value
}

const iconColors = computed(() => {
  const colors = {
    default: 'bg-surface-800 text-surface-400',
    accent: 'bg-accent-500/20 text-accent-400',
    danger: 'bg-rose-500/20 text-rose-400',
    warning: 'bg-amber-500/20 text-amber-400',
    success: 'bg-emerald-500/20 text-emerald-400',
  }
  return colors[props.variant || 'default']
})

const trendIcon = computed(() => {
  if (props.trend === 'up') return ArrowTrendingUpIcon
  if (props.trend === 'down') return ArrowTrendingDownIcon
  return MinusIcon
})
</script>

<template>
  <div class="card-dark p-5 hover:shadow-card-hover transition-all duration-300 group glow-border">
    <div class="flex items-start justify-between">
      <div class="flex-1 min-w-0">
        <p class="text-sm font-medium text-surface-400 truncate">{{ label }}</p>
        <p class="mt-2 text-2xl font-semibold text-surface-50 font-mono tracking-tight whitespace-nowrap">{{ formatValue(value) }}</p>
        <div v-if="subValue || trend" class="mt-2 flex items-center gap-2">
          <div
            v-if="trend"
            :class="[
              'flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full',
              trend === 'up' ? 'bg-emerald-500/20 text-emerald-400' : '',
              trend === 'down' ? 'bg-rose-500/20 text-rose-400' : '',
              trend === 'neutral' ? 'bg-surface-700 text-surface-400' : ''
            ]"
          >
            <component :is="trendIcon" class="w-3 h-3" />
          </div>
          <p v-if="subValue" class="text-sm text-surface-500 truncate">{{ subValue }}</p>
        </div>
      </div>

      <!-- Icon -->
      <div
        v-if="icon"
        :class="[iconColors, 'w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0 ml-4']"
      >
        <component :is="icon" class="w-6 h-6" />
      </div>

      <!-- Circular Progress (optional) -->
      <div v-if="progress !== undefined" class="relative w-14 h-14 flex-shrink-0 ml-4">
        <svg class="w-14 h-14 transform -rotate-90" viewBox="0 0 36 36">
          <path
            class="text-surface-800"
            stroke="currentColor"
            stroke-width="3"
            fill="none"
            d="M18 2.0845
              a 15.9155 15.9155 0 0 1 0 31.831
              a 15.9155 15.9155 0 0 1 0 -31.831"
          />
          <path
            class="text-accent-500"
            stroke="currentColor"
            stroke-width="3"
            fill="none"
            stroke-linecap="round"
            :stroke-dasharray="`${progress}, 100`"
            d="M18 2.0845
              a 15.9155 15.9155 0 0 1 0 31.831
              a 15.9155 15.9155 0 0 1 0 -31.831"
          />
        </svg>
        <span class="absolute inset-0 flex items-center justify-center text-xs font-semibold text-surface-200 font-mono">
          {{ progress }}%
        </span>
      </div>
    </div>
  </div>
</template>
