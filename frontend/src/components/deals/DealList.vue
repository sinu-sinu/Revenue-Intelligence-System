<script setup lang="ts">
import { useRouter } from 'vue-router'
import DataTable from '@/components/common/DataTable.vue'
import RiskBadge from '@/components/common/RiskBadge.vue'
import type { DealSummary } from '@/api/types'

defineProps<{
  deals: DealSummary[]
  loading?: boolean
  sortBy?: string
  sortOrder?: 'asc' | 'desc'
}>()

const emit = defineEmits<{
  sort: [column: string]
}>()

const router = useRouter()

const columns = [
  { key: 'opportunity_id', label: 'Deal ID', sortable: true },
  { key: 'account', label: 'Account', sortable: true },
  { key: 'product', label: 'Product', sortable: true },
  { key: 'sales_agent', label: 'Sales Rep', sortable: true },
  { key: 'risk_category', label: 'Risk', sortable: true },
  { key: 'risk_score', label: 'Score', sortable: true, align: 'center' as const },
  {
    key: 'win_probability',
    label: 'Win Prob',
    sortable: true,
    align: 'right' as const,
    format: (v: number) => `${(v * 100).toFixed(0)}%`,
  },
  {
    key: 'product_sales_price',
    label: 'Value',
    sortable: true,
    align: 'right' as const,
    format: (v: number) => `$${v.toLocaleString()}`,
  },
]

function handleRowClick(deal: DealSummary) {
  router.push(`/deals/${deal.opportunity_id}`)
}

function handleSort(column: string) {
  emit('sort', column)
}
</script>

<template>
  <DataTable
    :columns="columns"
    :data="deals"
    :loading="loading"
    :sort-by="sortBy"
    :sort-order="sortOrder"
    @sort="handleSort"
    @row-click="handleRowClick"
  >
    <template #cell-risk_category="{ value }">
      <RiskBadge :category="value" size="sm" />
    </template>
    <template #cell-risk_score="{ value, row }">
      <span
        :class="[
          'font-mono font-medium',
          {
            'text-rose-400': row.risk_category === 'Critical',
            'text-orange-400': row.risk_category === 'High',
            'text-amber-400': row.risk_category === 'Medium',
            'text-emerald-400': row.risk_category === 'Low',
          }
        ]"
      >
        {{ value }}
      </span>
    </template>
  </DataTable>
</template>
