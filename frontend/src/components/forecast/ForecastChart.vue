<script setup lang="ts">
import { computed } from 'vue'
import { Line } from 'vue-chartjs'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js'
import type { ForecastPeriod } from '@/api/types'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler)

const props = defineProps<{
  periods: ForecastPeriod[]
}>()

const chartData = computed(() => ({
  labels: props.periods.map((p) => p.period),
  datasets: [
    {
      label: 'P90 (Optimistic)',
      data: props.periods.map((p) => p.p90),
      borderColor: 'rgba(52, 211, 153, 0.8)',
      backgroundColor: 'rgba(52, 211, 153, 0.1)',
      fill: '+1',
      tension: 0.4,
      pointRadius: 0,
    },
    {
      label: 'P50 (Expected)',
      data: props.periods.map((p) => p.p50),
      borderColor: 'rgba(20, 184, 166, 1)',
      backgroundColor: 'rgba(20, 184, 166, 0.1)',
      fill: '+1',
      tension: 0.4,
      borderWidth: 3,
      pointRadius: 4,
      pointBackgroundColor: 'rgba(20, 184, 166, 1)',
    },
    {
      label: 'P10 (Pessimistic)',
      data: props.periods.map((p) => p.p10),
      borderColor: 'rgba(244, 63, 94, 0.8)',
      backgroundColor: 'rgba(244, 63, 94, 0.1)',
      fill: 'origin',
      tension: 0.4,
      pointRadius: 0,
    },
  ],
}))

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      position: 'top' as const,
      labels: {
        color: '#94a3b8',
        usePointStyle: true,
        padding: 20,
        font: {
          family: '"DM Sans", system-ui, sans-serif',
          size: 12,
        },
      },
    },
    tooltip: {
      backgroundColor: '#1e293b',
      titleColor: '#f1f5f9',
      bodyColor: '#94a3b8',
      borderColor: '#334155',
      borderWidth: 1,
      padding: 12,
      boxPadding: 4,
      usePointStyle: true,
      titleFont: {
        family: '"DM Sans", system-ui, sans-serif',
      },
      bodyFont: {
        family: '"JetBrains Mono", monospace',
      },
      callbacks: {
        label: (context: any) => {
          const value = context.raw as number
          return `${context.dataset.label}: $${value.toLocaleString()}`
        },
      },
    },
  },
  scales: {
    x: {
      grid: {
        color: 'rgba(51, 65, 85, 0.5)',
      },
      ticks: {
        color: '#64748b',
        font: {
          family: '"DM Sans", system-ui, sans-serif',
          size: 11,
        },
      },
    },
    y: {
      grid: {
        color: 'rgba(51, 65, 85, 0.5)',
      },
      ticks: {
        color: '#64748b',
        font: {
          family: '"JetBrains Mono", monospace',
          size: 11,
        },
        callback: (value: string | number) => `$${(Number(value) / 1000).toFixed(0)}K`,
      },
    },
  },
  interaction: {
    intersect: false,
    mode: 'index' as const,
  },
}
</script>

<template>
  <div class="h-80">
    <Line :data="chartData" :options="chartOptions" />
  </div>
</template>
