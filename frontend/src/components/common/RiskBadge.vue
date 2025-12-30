<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  category: 'Low' | 'Medium' | 'High' | 'Critical'
  score?: number
  size?: 'sm' | 'md' | 'lg'
}>()

const badgeClass = computed(() => {
  const base = 'inline-flex items-center font-medium rounded-full'
  const sizeClass = {
    sm: 'px-2.5 py-0.5 text-xs',
    md: 'px-3 py-1 text-sm',
    lg: 'px-4 py-1.5 text-base',
  }[props.size || 'md']

  const colorClass = {
    Critical: 'bg-rose-500/20 text-rose-400 ring-1 ring-inset ring-rose-500/30',
    High: 'bg-orange-500/20 text-orange-400 ring-1 ring-inset ring-orange-500/30',
    Medium: 'bg-amber-500/20 text-amber-400 ring-1 ring-inset ring-amber-500/30',
    Low: 'bg-emerald-500/20 text-emerald-400 ring-1 ring-inset ring-emerald-500/30',
  }[props.category]

  return `${base} ${sizeClass} ${colorClass}`
})

const dotClass = computed(() => {
  return {
    Critical: 'bg-rose-500',
    High: 'bg-orange-500',
    Medium: 'bg-amber-500',
    Low: 'bg-emerald-500',
  }[props.category]
})
</script>

<template>
  <span :class="badgeClass">
    <span :class="[dotClass, 'w-1.5 h-1.5 rounded-full mr-1.5']" />
    {{ category }}
    <span v-if="score !== undefined" class="ml-1.5 font-mono font-normal opacity-75">{{ score }}</span>
  </span>
</template>
